#include <astra/astra.hpp>
#include <cstdio>
#include <iostream>

class SampleFrameListener : public astra::FrameListener
{
private:
    using buffer_ptr = std::unique_ptr< astra::RgbPixel[] >;
    buffer_ptr buffer_;
    astra::DepthFrame *_depthFrame;
    astra::ColorFrame *_colorFrame;

public:
	//SampleFrameListener(){}

    virtual void on_frame_ready( astra::StreamReader &reader,
        astra::Frame &frame ) override
    {
        _colorFrame = new astra::ColorFrame( frame.get< astra::ColorFrame >() );
        _depthFrame = new astra::DepthFrame( frame.get< astra::DepthFrame >() );
    }

    auto getColorFrame() const { return *_colorFrame; }
    auto getDepthFrame() const { return *_depthFrame; }
};

class AstraStream
{
private:
    SampleFrameListener _listener;

public:
	AstraStream(){}
	
    void init()
    {
        astra::initialize();
        astra::StreamSet streamSet;
        astra::StreamReader reader = streamSet.create_reader();
        reader.stream< astra::ColorStream >().start();
        reader.add_listener( _listener );
    }

    void proceedFrame()
    {
        astra_update();
    }

    astra::ColorFrame getRGB()
    {
        return _listener.getColorFrame();
    }

    astra::DepthFrame getDepth()
    {
        return _listener.getDepthFrame();
    }

    void close()
    {
        astra::terminate();
    }
};