import threading
import time
import os
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from driver_monitor import DriverMonitoringSystem

app = FastAPI(title="DriveWise API Server")

# Configure CORS so the React frontend can poll this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared global monitor instance
monitor = None
monitor_thread = None

def start_monitor():
    global monitor
    print("[Backend] Initializing Driver Monitoring System...")
    monitor = DriverMonitoringSystem(headless=True)
    try:
        monitor.run()
    except Exception as e:
        print(f"[Backend] Error running monitor loop: {e}")

@app.on_event("startup")
def startup_event():
    global monitor_thread
    monitor_thread = threading.Thread(target=start_monitor, daemon=True)
    monitor_thread.start()
    print("[Backend] Background camera monitoring thread started.")

@app.on_event("shutdown")
def shutdown_event():
    global monitor
    if monitor is not None:
        print("[Backend] Shutting down monitor...")
        monitor.is_running = False

@app.get("/api/driver-status")
def get_driver_status():
    global monitor
    if monitor is None or not monitor.current_telemetry:
        # Initial default safe values while camera initializes
        return {
            "risk_score": 0,
            "risk_level": "SAFE",
            "speed": 0,
            "ear": 0.28,
            "emotion": "neutral",
            "phone_detected": False,
            "passenger_detected": False,
            "talking_detected": False,
            "driver_distracted": False,
            "alerts": []
        }
    return monitor.current_telemetry

def gen_video_frames():
    global monitor
    while True:
        if monitor is not None and monitor.current_frame is not None:
            # Encode frame to JPEG format
            ret, jpeg = cv2.imencode('.jpg', monitor.current_frame)
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Small throttle to not overload CPU
        time.sleep(0.04)

@app.get("/api/video-feed")
def get_video_feed():
    return StreamingResponse(
        gen_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    # Run server on port 8000
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
