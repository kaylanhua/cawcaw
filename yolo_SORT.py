import sys
import traceback
import tellopy
import av
import cv2 as cv2  # for avoidance of pylint error
import numpy
import time

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

def main():
    drone = tellopy.Tello()

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        container = av.open(drone.get_video_stream())
        # skip first 300 frames
        frame_skip = 300
        iter = 0
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                # cv2.imwrite('/Users/mehulgoel/Documents/DATA/Video1/' + str(iter) + '.jpg', image)

                if iter % 10 == 0:
                    results = model.track(image, persist=True)
                    annotated_frame = results[0].plot()
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)
                    
                cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                cv2.imshow('Video', image)
                cv2.waitKey(1)
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)

                iter += 1
                    

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
