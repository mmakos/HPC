from datetime import datetime
from primesense import openni2
from primesense import _openni2 as c_api
import sys
import keyboard
import argparse


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument( "-v", "--video_path", default=datetime.now().strftime( "%Y%m%d%H%M%S" ), help="Path to recorded video relative to /data/video." )
    return parser.parse_known_args()[ 0 ]


def record():
    depthStream = dev.create_depth_stream()
    colorStream = dev.create_color_stream()
    depthStream.set_video_mode( c_api.OniVideoMode( pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                    resolutionX = 640, resolutionY = 480, fps = 30 ) )
    colorStream.set_video_mode( c_api.OniVideoMode( pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                    resolutionX = 1280, resolutionY = 720, fps = 30 ) )
    dev.set_image_registration_mode( True )
    dev.set_depth_color_sync_enabled( True )
    depthStream.set_mirroring_enabled( True )
    colorStream.set_mirroring_enabled( True )

    depthStream.start()
    colorStream.start()

    args = getArgs()
    fileName = "../../data/video/" + args.video_path

    recorder = openni2.Recorder( ( fileName + ".oni" ).encode( 'utf-8' ) )
    recorder.attach( depthStream )
    recorder.attach( colorStream )

    print( "Press 's' to start recording" )
    keyboard.wait( "s" )
    print()

    recorder.start()

    print( "Recording...\nPress 'q' to stop recording" )
    keyboard.wait( "q" )
    print()

    recorder.stop()
    depthStream.stop()
    colorStream.stop()


if __name__ == '__main__':
    if sys.platform == "win32":
        openni2.initialize(
            "../../externals/OpenNI/Windows/Astra OpenNI2 Development Instruction(x64)_V1.3/OpenNI2/OpenNI-Windows-x64-2.3.0.63/Redist" )
    else:
        openni2.initialize( "../../OpenNI/Linux/OpenNI-Linux-x64-2.3.0.63/Redist" )
    print( "Device initialized." )

    dev = openni2.Device.open_any()
    record()

    openni2.unload()
    print( "Device unloaded." )
