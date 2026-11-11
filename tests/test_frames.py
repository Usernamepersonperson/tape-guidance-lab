import pytest

from agvlab import frames


def test_cob_ids_for_default_node():
    ids = frames.cob_ids(frames.DEFAULT_NODE_ID)
    assert frames.DEFAULT_NODE_ID == 21
    assert ids == {"tpdo1": 0x195, "tpdo2": 0x295, "tpdo3": 0x395, "heartbeat": 0x715}


@pytest.mark.parametrize("node", [0, 128, -1])
def test_cob_ids_reject_bad_node(node):
    with pytest.raises(ValueError):
        frames.cob_ids(node)


def test_tpdo1_golden_vector():
    f = frames.TrackFrame.from_physical(8.0, -12.3, left_deg=1.0, right_deg=-2.5, quality_pct=100.0, fork=True)
    assert frames.encode_tpdo1(f).hex() == "500085ff02fb07c8"


def test_decode_tpdo1_golden_vector_physical_values():
    f = frames.decode_tpdo1(bytes.fromhex("500085ff02fb07c8"))
    assert f.left_offset_mm == pytest.approx(8.0) and f.right_offset_mm == pytest.approx(-12.3)
    assert f.left_heading_deg == 1.0 and f.right_heading_deg == -2.5 and f.quality_pct == 100.0
    assert f.tape_detected and f.two_tracks and f.fork and not (f.merge or f.junction or f.fault)


def test_flag_bit_layout():
    one = {"detected": 0x01, "two_tracks": 0x02, "fork": 0x04, "merge": 0x08, "junction": 0x10,
           "marker_left": 0x20, "marker_right": 0x40, "fault": 0x80}
    for name, bit in one.items():
        data = frames.encode_tpdo1(frames.TrackFrame(**{name: True}))
        assert data[6] == bit, name


def test_no_tape_frame_is_all_sentinels():
    data = frames.encode_tpdo1(frames.TrackFrame())
    assert data.hex() == "0080008000000000"
    f = frames.decode_tpdo1(data)
    assert not f.tape_detected and f.left_offset_mm is None and f.right_offset_mm is None


def test_roundtrip_all_flags():
    f = frames.TrackFrame(-123, 77, 3, -4, 150, True, True, True, False, True, True, False, True)
    assert frames.decode_tpdo1(frames.encode_tpdo1(f)) == f


def test_offsets_are_little_endian_twos_complement():
    assert frames.encode_tpdo1(frames.TrackFrame(-1, 256, 0, 0))[:4] == bytes([0xFF, 0xFF, 0x00, 0x01])


def test_fields_saturate_instead_of_wrapping():
    f = frames.decode_tpdo1(frames.encode_tpdo1(frames.TrackFrame(40000, -40000, 500, -500, 900, detected=True)))
    assert (f.left_offset_raw, f.right_offset_raw) == (32767, -32767)
    assert (f.left_heading_raw, f.right_heading_raw) == (127, -128)
    assert f.quality_raw == 200


def test_no_data_sentinel_is_never_produced_by_saturation():
    assert frames.decode_tpdo1(frames.encode_tpdo1(frames.TrackFrame(-32767, 0, 0, 0, 0))).left_offset_raw == -32767


def test_from_physical_rounds_to_wire_resolution():
    f = frames.TrackFrame.from_physical(1.26, left_deg=0.6, quality_pct=99.8)
    assert f.left_offset_raw == 13 and f.left_heading_raw == 1 and f.quality_raw == 200
    assert f.right_offset_raw == 13 and not f.two_tracks and f.detected  # single tape: both slots equal


def test_from_physical_none_means_no_tape():
    assert frames.TrackFrame.from_physical(None) == frames.TrackFrame()


def test_single_track_property():
    assert frames.TrackFrame.from_physical(4.0).single_track
    assert not frames.TrackFrame.from_physical(4.0, 5.0).single_track


def test_tpdo2_golden_vector_and_scaling():
    m = frames.MarkerFrame(left_along_raw=100, left_lateral_raw=-1100)
    data = frames.encode_tpdo2(m)
    assert data.hex() == "6400b4fb00800080"
    d = frames.decode_tpdo2(data)
    assert d.left_along_mm == pytest.approx(5.0) and d.left_lateral_mm == pytest.approx(-55.0)
    assert d.right_along_mm is None and d.right_lateral_mm is None


def test_tpdo3_layout():
    data = frames.encode_tpdo3(frames.DiagFrame(0x1234, 1))
    assert data == bytes([0x34, 0x12, 0x01])
    assert frames.decode_tpdo3(data) == frames.DiagFrame(0x1234, 1)


def test_tpdo3_counter_wraps_at_16_bits():
    assert frames.decode_tpdo3(frames.encode_tpdo3(frames.DiagFrame(65537, 0))).sample_counter == 1


@pytest.mark.parametrize("fn,n", [(frames.decode_tpdo1, 8), (frames.decode_tpdo2, 8), (frames.decode_tpdo3, 3)])
def test_wrong_length_rejected(fn, n):
    with pytest.raises(frames.DecodeError):
        fn(bytes(n + 1))
    with pytest.raises(frames.DecodeError):
        fn(bytes(n - 1))


def test_decode_frame_dispatch_and_node_filter():
    t1 = frames.encode_tpdo1(frames.TrackFrame.from_physical(1.0))
    assert isinstance(frames.decode_frame(0x195, t1, 21), frames.TrackFrame)
    assert frames.decode_frame(0x196, t1, 21) is None  # other node
    assert frames.decode_frame(0x595, bytes(8), 21) is None  # SDO response: not decoded here
    assert frames.decode_frame(0x715, b"\x05", 21) == frames.HeartbeatFrame(5)
    assert frames.decode_frame(0x1195, t1, 21) is None  # not an 11-bit id
    with pytest.raises(frames.DecodeError):
        frames.decode_frame(0x195, b"\x00", 21)
    with pytest.raises(frames.DecodeError):
        frames.decode_frame(0x715, b"\x05\x00", 21)


def test_pdo_mappings_fill_the_dlc_exactly():
    for name, entries in frames.PDO_MAPPING.items():
        assert sum(bits for _i, _s, bits in entries) == frames.PDO_DLC[name] * 8, name


def test_pdo_mappings_point_at_declared_objects_with_matching_size():
    declared = {(i, s): bits for i, s, _n, _t, bits, _a, _d in frames.OBJECTS}
    for name, entries in frames.PDO_MAPPING.items():
        for index, sub, bits in entries:
            assert declared[(index, sub)] == bits, (name, hex(index), sub)


def test_mapping_entry_encoding():
    assert frames.mapping_entry(0x2000, 1, 16) == 0x20000110


def test_struct_sizes_match_declared_dlc():
    assert frames._T1.size == 8 and frames._T2.size == 8 and frames._T3.size == 3


def test_eds_text_is_valid_ini_and_lists_every_object():
    import configparser
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read_string(frames.eds_text())
    assert "2000sub1" in cp and "1A00sub6" in cp and "1017" in cp
    assert cp["1A00sub1"]["DefaultValue"] == "0x20000110"
    assert cp["2000"]["SubNumber"] == "7"
