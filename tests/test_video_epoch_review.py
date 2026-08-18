import pandas as pd

from dash_app.app import _epoch_rows_for_selection


def test_epoch_review_uses_layer1_geometry_not_stale_final_geometry():
    rec = {
        "layer1": pd.DataFrame({
            "epoch_id": [0, 1, 2],
            "t0_s": [9900.0, 9901.0, 9902.0],
            "t1_s": [9901.0, 9902.0, 9903.0],
            "layer1_label": ["Wake", "Sleep", "Sleep"],
        }),
        # Deliberately stale/non-overlapping Final geometry. This must not make
        # the synchronized reviewer claim there are no scoring epochs.
        "final": pd.DataFrame({
            "t0_s": [0.0],
            "t1_s": [1.0],
            "final_state": ["Wake"],
        }),
    }
    selected = {"start_min": 165.0, "end_min": 165.05}  # 9900–9903 s
    epochs = _epoch_rows_for_selection(rec, selected)
    assert len(epochs) == 3
    assert epochs["t0_s"].tolist() == [9900.0, 9901.0, 9902.0]
    assert set(epochs["final_state"]) == {"Undefined"}


def test_local_qc_clip_reuse_requires_same_source_and_coverage(tmp_path):
    from dash_app.app import _clip_info_covers_selection, _source_video_signature

    source = tmp_path / "long_video.mp4"
    source.write_bytes(b"source-video")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"local-qc-clip")

    info = {
        "clip_path": str(clip),
        "source_signature": _source_video_signature(source),
        "source_start_s": 100.0,
        "source_end_s": 220.0,
    }
    assert _clip_info_covers_selection(info, source, 120.0, 180.0)
    assert not _clip_info_covers_selection(info, source, 90.0, 180.0)
    assert not _clip_info_covers_selection(info, source, 120.0, 230.0)

    source.write_bytes(b"source-video-changed")
    assert not _clip_info_covers_selection(info, source, 120.0, 180.0)


def test_cached_epoch_review_player_is_explicitly_marked_local(tmp_path):
    from dash_app.app import epoch_review_cached_video_children

    clip = tmp_path / "short_qc.mp4"
    clip.write_bytes(b"not-a-real-video-but-present")
    component = epoch_review_cached_video_children({
        "clip_path": str(clip),
        "source_start_s": 120.0,
        "source_end_s": 190.0,
    })
    video = component.children[0]
    props = video.to_plotly_json()["props"]
    assert props["id"] == "epoch-review-video-player"
    assert props["data-qc-local-clip"] == "1"
    assert props["data-source-start-s"] == "120.000000"
    assert props["data-source-end-s"] == "190.000000"
