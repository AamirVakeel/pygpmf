from collections import namedtuple
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

import gpxpy
from . import parse
import numpy as np
import struct


GPSData = namedtuple("GPSData",
                     [
                         "description",
                         "timestamp",
                         "precision",
                         "fix",
                         "latitude",
                         "longitude",
                         "altitude",
                         "speed_2d",
                         "speed_3d",
                         "units",
                         "npoints"
                     ])


def extract_gps_blocks(stream):
    """ Extract GPS data blocks from binary stream

    This is a generator on lists `KVLItem` objects. In
    the GPMF stream, GPS data comes into blocks of several
    different data items. For each of these blocks we return a list.

    Parameters
    ----------
    stream: bytes
        The raw GPMF binary stream

    Returns
    -------
    gps_items_generator: generator
        Generator of lists of `KVLItem` objects
    """
    for s in parse.filter_klv(stream, "STRM"):
        content = []
        is_gps = False
        for elt in s.value:
            content.append(elt)
            if elt.key == "GPS9":
                is_gps = True
        if is_gps:
            yield content

# Function to decode a single 32-byte GPS9 entry
def parse_gps9_payload_block(block_bytes, scalers):
    # if len(block_bytes) != 32:
    #     raise ValueError("Block must be exactly 32 bytes")

    # unpacked = struct.unpack(">iibbhhhhhhhhh", block_bytes)
    (
        lat, lon, alt,
        speed2d, speed3d,
        days, secs,
        dop, fix
    ) = struct.unpack(">iiiiiiihh", block_bytes)
    return {
        "latitude": lat / scalers[0],
        "longitude": lon / scalers[1],
        "altitude": alt / scalers[2],
        "speed2d": speed2d / scalers[3],
        "speed3d": speed3d / scalers[4],
        "days": days / scalers[5],
        "secs": secs / scalers[6],
        "dop": dop / scalers[7],
        "fix": fix / scalers[8],
    }

# Function to process entire GPS9 binary stream
def parse_gps9_data(binary_data, scalers):
    block_size = 32
    num_blocks = len(binary_data) // block_size
    data = []

    for i in range(num_blocks):
        block = binary_data[i*block_size:(i+1)*block_size]
        data.append(parse_gps9_payload_block(block[:32], scalers))

    return data

def calculate_date(days, seconds):
    j2000_epoch = datetime(2000, 1, 1, 0, 0, 0)
    final_time = j2000_epoch + timedelta(days=days, seconds=seconds)
    return final_time.strftime('%Y-%m-%d %H:%M:%S.%f')

def parse_gps_block(gps_block):
    """Turn GPS data blocks into `GPSData` objects

    Parameters
    ----------
    gps_block: list of KVLItem
        A list of KVLItem corresponding to a GPS data block.

    Returns
    -------
    gps_data: GPSData
        A GPSData object holding the GPS information of a block.
    """
    block_dict = {
        s.key: s for s in gps_block
    }

    gps_data_object = parse_gps9_data(block_dict.get("GPS9").value, block_dict.get("SCAL").value)

    if gps_data_object is not None:
        # latitude, longitude, altitude, speed_2d, speed_3d = gps_data.T
        latitude = np.array([float(each["latitude"]) for each in gps_data_object])
        longitude = np.array([float(each["longitude"]) for each in gps_data_object])
        altitude = np.array([float(each["altitude"]) for each in gps_data_object])
        speed_2d = np.array([float(each["speed2d"]) for each in gps_data_object])
        speed_3d = np.array([float(each["speed3d"]) for each in gps_data_object])
        days = np.array([each["days"] for each in gps_data_object])
        secs = np.array([each["secs"] for each in gps_data_object])
        dop = np.array([each["dop"] for each in gps_data_object])
        fix = np.array([each["fix"] for each in gps_data_object])

        return GPSData(
            description=block_dict["STNM"].value,
            timestamp=calculate_date(days[0], secs[0]),
            precision=dop[0],
            fix=fix[0],
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            speed_2d=speed_2d,
            speed_3d=speed_3d,
            units=block_dict["UNIT"].value[:5],
            npoints=len(gps_data_object)
        )
    else:
        return None


FIX_TYPE = {
    0: "none",
    2: "2d",
    3: "3d"
}


def _make_speed_extensions(gps_data, i = None):
    speed_2d = ET.Element("speed_2d")
    value = ET.SubElement(speed_2d, "value")
    value.text = ("%g" % gps_data.speed_2d) if i is None else ("%g" % gps_data.speed_2d[i])
    unit = ET.SubElement(speed_2d, "unit")
    unit.text = "m/s"

    speed_3d = ET.Element("speed_3d")
    value = ET.SubElement(speed_3d, "value")
    value.text = ("%g" % gps_data.speed_3d)  if i is None else ("%g" % gps_data.speed_3d[i])
    unit = ET.SubElement(speed_3d, "unit")
    unit.text = "m/s"

    return [speed_2d, speed_3d]


def make_pgx_segment(gps_blocks, first_only=False, speeds_as_extensions=True):
    """Convert a list of GPSData objects into a GPX track segment.

    Parameters
    ----------
    gps_blocks: list of GPSData
        A list of GPSData objects
    first_only: bool, optional (default=False)
        If True use only the first GPS entry of each data block.
    speeds_as_extensions: bool, optional (default=True)
        If True, include 2d and 3d speed values as exentensions of
        the GPX trackpoints. This is especially useful when saving
        to GPX 1.1 format.

    Returns
    -------
    gpx_segment: gpxpy.gpx.GPXTrackSegment
        A gpx track segment.
    """

    track_segment = gpxpy.gpx.GPXTrackSegment()
    dt = timedelta(seconds=1.0 / 18.)

    for gps_data in gps_blocks:
        if gps_data is not None:
            time = datetime.strptime(gps_data.timestamp, "%Y-%m-%d %H:%M:%S.%f")
            # Reference says the frequency is about 18 Hz and other GPS data about 1Hz

            if((type(gps_data.latitude) != type(np.array([]))) and (type(gps_data.longitude) != type(np.array([])))):
                tp = gpxpy.gpx.GPXTrackPoint(
                    latitude=gps_data.latitude,
                    longitude=gps_data.longitude,
                    elevation=gps_data.altitude,
                    speed=gps_data.speed_3d,
                    position_dilution=gps_data.precision,
                    time=time,
                    symbol="Square",
                )

                tp.type_of_gpx_fix = FIX_TYPE[gps_data.fix]

                if speeds_as_extensions:

                    for e in _make_speed_extensions(gps_data, None):
                        tp.extensions.append(e)

                track_segment.points.append(tp)
                continue

            # stop = 1 if first_only else gps_data.npoints
            stop = min([len(gps_data.latitude),len(gps_data.longitude)])
            for i in range(stop):
                tp = gpxpy.gpx.GPXTrackPoint(
                    latitude=gps_data.latitude[i],
                    longitude=gps_data.longitude[i],
                    elevation=gps_data.altitude[i],
                    speed=gps_data.speed_3d[i],
                    position_dilution=gps_data.precision,
                    time=time + i * dt,
                    symbol="Square",
                )

                tp.type_of_gpx_fix = FIX_TYPE[gps_data.fix]

                if speeds_as_extensions:

                    for e in _make_speed_extensions(gps_data, 0):
                        tp.extensions.append(e)

                track_segment.points.append(tp)

    return track_segment
