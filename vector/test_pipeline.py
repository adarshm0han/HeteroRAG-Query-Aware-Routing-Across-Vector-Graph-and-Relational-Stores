import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pipeline

from pipeline import MAX_TABLE_IDS, load_passages, record_id, to_metadata


def test_load_passages_dedupes_and_tracks_tables(tmp_path):
    (tmp_path / 'Table_A_0.json').write_text(json.dumps({'/wiki/X': 'X is a thing .', '/wiki/Y': ' '}))
    (tmp_path / 'Table_B_0.json').write_text(json.dumps({'/wiki/X': 'X is a thing .'}))
    passages = load_passages(tmp_path)
    assert [p['passage_id'] for p in passages] == ['/wiki/X']
    assert passages[0]['title'] == 'X'
    assert sorted(passages[0]['table_ids']) == ['Table_A_0', 'Table_B_0']


def test_record_id_is_ascii_and_stable():
    rid = record_id('/wiki/São_Paulo')
    assert rid.isascii() and rid == record_id('/wiki/São_Paulo')
    assert record_id("/wiki/D.C._United") == "/wiki/D.C._United"


def test_metadata_caps_table_ids():
    meta = to_metadata({'passage_id': '/wiki/U', 'title': 'U', 'text': 't', 'table_ids': [str(i) for i in range(900)]})
    assert len(meta['table_ids']) == MAX_TABLE_IDS and meta['table_count'] == 900


def test_download_archive_preserves_windows_unsafe_table_ids(tmp_path, monkeypatch):
    source = tmp_path / 'source.zip'
    table_id = '"Weird_Al"_Yankovic_0'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr(f'WikiTables-WithLinks-{pipeline.PASSAGE_REVISION}/request_tok/{table_id}.json',
                         json.dumps({'/wiki/X': 'A passage.'}))
    data = tmp_path / 'data'
    monkeypatch.setattr(pipeline, 'DATA', data)
    monkeypatch.setattr(pipeline, 'ARCHIVE', data / 'corpus.zip')
    monkeypatch.setattr(pipeline, 'PASSAGE_SHA256', pipeline.file_sha256(source))
    pipeline.cmd_download(SimpleNamespace(archive=source))
    passages = load_passages(archive_path=pipeline.ARCHIVE)
    assert passages[0]['table_ids'] == [table_id]
    assert sorted(p.name for p in data.iterdir()) == ['corpus.zip']


def test_download_rejects_corrupt_archive(tmp_path, monkeypatch):
    source = tmp_path / 'bad.zip'
    source.write_bytes(b'not the pinned archive')
    monkeypatch.setattr(pipeline, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(pipeline, 'ARCHIVE', pipeline.DATA / 'corpus.zip')
    with pytest.raises(ValueError, match='checksum mismatch'):
        pipeline.cmd_download(SimpleNamespace(archive=source))
    assert not pipeline.ARCHIVE.exists()


@pytest.fixture
def upload_fixture(tmp_path, monkeypatch):
    for name, filename in [('EMBEDDINGS', 'embeddings.npy'), ('PASSAGES', 'passages.parquet'),
                           ('CHECKPOINT', 'upload_checkpoint.json')]:
        monkeypatch.setattr(pipeline, name, tmp_path / filename)
    passages = [{'passage_id': f'/wiki/{i}', 'title': str(i), 'text': 'sample', 'table_ids': ['Table_0']}
                for i in range(3)]
    np.save(pipeline.EMBEDDINGS, np.zeros((3, pipeline.DIM), dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(passages), pipeline.PASSAGES)
    calls = []
    index = SimpleNamespace(config=SimpleNamespace(host='https://first-index.example'),
                            upsert=lambda *, vectors: calls.append(vectors))
    monkeypatch.setattr(pipeline, 'get_index', lambda create=False: index)
    monkeypatch.setattr(pipeline, 'UPSERT_BATCH', 2)
    return index, calls


def test_upload_resumes_after_trial_limit(upload_fixture):
    _, calls = upload_fixture
    pipeline.cmd_upload(SimpleNamespace(limit=1))
    pipeline.cmd_upload(SimpleNamespace(limit=None))
    assert [v['id'] for batch in calls for v in batch] == ['/wiki/0', '/wiki/1', '/wiki/2']
    assert json.loads(pipeline.CHECKPOINT.read_text())['next'] == 3
    pipeline.cmd_upload(SimpleNamespace(limit=None))
    assert len(calls) == 2


@pytest.mark.parametrize('change', ['index', 'passages', 'embeddings', 'legacy_checkpoint'])
def test_upload_rejects_checkpoint_for_other_inputs(upload_fixture, change):
    index, calls = upload_fixture
    pipeline.cmd_upload(SimpleNamespace(limit=1))
    if change == 'index':
        index.config.host = 'https://second-index.example'
    elif change == 'passages':
        passages = pq.read_table(pipeline.PASSAGES).to_pylist()
        passages[0]['text'] = 'changed text'
        pq.write_table(pa.Table.from_pylist(passages), pipeline.PASSAGES)
    elif change == 'embeddings':
        np.save(pipeline.EMBEDDINGS, np.ones((3, pipeline.DIM), dtype=np.float32))
    else:
        pipeline.CHECKPOINT.write_text(json.dumps({'next': 1}))
    with pytest.raises(ValueError, match='another index/dataset'):
        pipeline.cmd_upload(SimpleNamespace(limit=None))
    assert len(calls) == 1


def test_failed_batch_is_retried_without_skipping_records(upload_fixture):
    index, calls = upload_fixture
    def fail_second_batch(*, vectors):
        if calls:
            raise RuntimeError('temporary service error')
        calls.append(vectors)
    index.upsert = fail_second_batch
    with pytest.raises(RuntimeError):
        pipeline.cmd_upload(SimpleNamespace(limit=None))
    assert json.loads(pipeline.CHECKPOINT.read_text())['next'] == 2
    index.upsert = lambda *, vectors: calls.append(vectors)
    pipeline.cmd_upload(SimpleNamespace(limit=None))
    assert [v['id'] for batch in calls for v in batch] == ['/wiki/0', '/wiki/1', '/wiki/2']
