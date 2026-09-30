#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for aspect-preserving max_side resize."""

from video_resize import resolve_max_side_size, resize_frame_max_side


def test_resolve_keeps_1080p_and_below():
    assert resolve_max_side_size(1920, 1080, 1920) == (1920, 1080)
    assert resolve_max_side_size(1080, 1920, 1920) == (1080, 1920)
    assert resolve_max_side_size(720, 480, 1920) == (720, 480)


def test_resolve_4k_and_8k_keep_aspect():
    assert resolve_max_side_size(3840, 2160, 1920) == (1920, 1080)
    assert resolve_max_side_size(2160, 3840, 1920) == (1080, 1920)
    assert resolve_max_side_size(7680, 4320, 1920) == (1920, 1080)
    assert resolve_max_side_size(1920, 1080, 1080) == (1080, 608)


def test_resolve_disabled():
    assert resolve_max_side_size(3840, 2160, 0) == (3840, 2160)
    assert resolve_max_side_size(3840, 2160, None) == (3840, 2160)


def test_resize_frame_noop_and_downscale():
    try:
        import cv2  # noqa: F401
        import numpy as np
    except ImportError:
        print("skip resize_frame_max_side (cv2 not installed)")
        return

    small = np.zeros((480, 720, 3), dtype=np.uint8)
    assert resize_frame_max_side(small, 1920).shape == (480, 720, 3)

    big = np.zeros((2160, 3840, 3), dtype=np.uint8)
    out = resize_frame_max_side(big, 1920)
    assert out.shape == (1080, 1920, 3)


if __name__ == "__main__":
    test_resolve_keeps_1080p_and_below()
    test_resolve_4k_and_8k_keep_aspect()
    test_resolve_disabled()
    test_resize_frame_noop_and_downscale()
    print("ok")
