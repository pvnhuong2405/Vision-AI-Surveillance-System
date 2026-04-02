import cv2
from ultralytics import YOLO
import time


model = YOLO("/Users/phamngochuong/Desktop/Vision AI Camera/Object_detection/Model/yolo11n.pt")  

cap = cv2.VideoCapture("/Users/phamngochuong/Desktop/Vision AI Camera/test1.mov")

def detect_tracking(people):
    people = {}
    TIMEOUT = 2  # giây

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # resize để tăng tốc
        # frame = cv2.resize(frame, (640, 360))

        results = model.track(frame, persist=True, classes=[0])

        boxes = results[0].boxes

        current_time = time.time()

        if boxes.id is not None:
            for box, track_id in zip(boxes.xyxy, boxes.id):

                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)

                center_x = (x1 + x2)//2
                center_y = (y1 + y2)//2

                # INIT
                if track_id not in people:
                    people[track_id] = {
                        "start_time": current_time,
                        "last_seen": current_time,
                        "path": []
                    }

                # UPDATE
                people[track_id]["last_seen"] = current_time
                people[track_id]["path"].append((center_x, center_y))

                # TIME
                duration = current_time - people[track_id]["start_time"]
                minutes = int(duration // 60)
                seconds = int(duration % 60)

                label = f"ID {track_id} - {minutes}m {seconds}s"

                # DRAW BOX
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # DRAW TRAJECTORY - quỷ đạo 
                # path = people[track_id]["path"]
                # for i in range(1, len(path)):
                #     cv2.line(frame, path[i-1], path[i], (255,0,0), 2)

        # XÓA NGƯỜI BIẾN MẤT
        remove_ids = []
        for pid in people:
            if current_time - people[pid]["last_seen"] > TIMEOUT:
                remove_ids.append(pid)

        for pid in remove_ids:
            del people[pid]

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    
    cap.release()
    cv2.destroyAllWindows()
    return people

