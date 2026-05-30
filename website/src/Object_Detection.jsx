import styles from './Object_Detection.module.css';
import { useRef, useEffect, useState } from 'react'

function Object_Detection() {
    const videoRef = useRef(null);
    const canvasRefBBox = useRef(null);
    const canvasRefSending = useRef(null);
    const ctxRefBBox = useRef(null);
    const ctxRefSending = useRef(null);
    const [error, setError] = useState(false);
    const bboxes = useRef(null)
    const socketRef = useRef(null); 
    const [running, setRunning] = useState(false);

    useEffect(() => {
        if (canvasRefBBox.current) {
            ctxRefBBox.current = canvasRefBBox.current.getContext("2d");
            printDefault();
        }

        if (canvasRefSending.current) {
            ctxRefSending.current = canvasRefSending.current.getContext("2d");
        }
    }, []);

    useEffect(() => {
        async function startWebcam() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false,
                });

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (err) {
                console.error("Error accessing webcam:", err);
            }
        }

        startWebcam();

        return () => {
            if (videoRef.current?.srcObject) {
                const tracks = videoRef.current.srcObject.getTracks();
                tracks.forEach((track) => track.stop());
            }
        };
    }, []);

    const drawBBox = () => {
        const video = videoRef.current;
        const canvas = canvasRefBBox.current;

        const ctx = ctxRefBBox.current;
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = "red";
        ctx.lineWidth = 1;

        if (bboxes.current) {
            bboxes.current.forEach(edge => {
                const x1 = Math.max(edge[0], 0);
                const y1 = Math.max(edge[1], 0);
                const x2 = Math.min(edge[2], canvas.width);
                const y2 = Math.min(edge[3], canvas.height);
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            });
        }
    };

    const sendImage = () => {
        const video = videoRef.current;
        const canvas = canvasRefSending.current;

        const ctx = ctxRefSending.current;
        if (!ctx) return;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            if (!blob) return;

            try {
                socketRef.current?.send(blob)
            } catch (e) {
                setError(true);
            }
        }, "image/jpeg", 1);
    };

    const printDefault = () => {
        const ctx = ctxRefBBox.current;
        if (!ctx) return;

        const img = new Image();
        img.src = "/yolo.jfif";

        const canvas = canvasRefBBox.current;

        const canvasW = canvas.width;
        const canvasH = canvas.height;

        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
    }

    const startRecording = () => {
        if (running) {
            if (socketRef.current) {
                socketRef.current.close();
                socketRef.current = null;
            }

            bboxes.current = null;

            printDefault();
            setRunning(false);
            return;
        }

        const video = videoRef.current;
        const canvas = canvasRefBBox.current;

        const ctx = ctxRefBBox.current;
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const socket = new WebSocket("wss://api.marvinkrausser.com/predict_face");
        socketRef.current = socket;

        socket.onopen = () => {
            console.log("Connected");
            sendImage();
        }
        socket.onmessage = (event) => {
            bboxes.current = JSON.parse(event.data).bboxes;
            drawBBox();
            sendImage();
        }
        socket.onerror = console.error;
        socket.onclose = () => console.log("Disconnected");

        setRunning(true);
    }

    return (
        <div className='site-box'>
            <h1 className='site-headline'>Face Detection</h1>
            <div className={styles["button-div"]}>
                <button
                    id="button-send"
                    onClick={startRecording}
                    style={{ display: "none" }}
                />

                <label
                    htmlFor="button-send"
                    className="custom-button"
                >
                    {running? "End Detection" : "Start Detection"}
                </label>
            </div>
            <div id={styles["video-container"]}>
                <video autoPlay={true} ref={videoRef} style={{ width: "100%", height: "100%", position: "absolute", zIndex: "3" }}></video>
                <canvas ref={canvasRefBBox} style={{width: "100%", height: "100%", position: "absolute", zIndex: "4"}} />
                <canvas ref={canvasRefSending} style={{width: "100%", height: "100%", display: "none"}} />
            </div>

            <div id={styles["text-container"]}>
                <p>
                    This Project is an implementation of the YOLO (You Only Look Once) algorithm. It was implemented with Pytorch and trained on a Dataset from Roboflow.
                </p>
            </div>
        </div>
    )
}

export default Object_Detection;