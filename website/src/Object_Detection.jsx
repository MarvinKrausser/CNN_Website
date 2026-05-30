import styles from './Object_Detection.module.css';
import { useRef, useEffect, useState } from 'react'

function Object_Detection() {
    const videoRef = useRef(null);
    const canvasRefBBox = useRef(null);
    const canvasRefSending = useRef(null);
    const ctxRefBBox = useRef(null);
    const ctxRefSending = useRef(null);
    const [error, setError] = useState(false);
    const [loading, setLoading] = useState(false);
    const bboxes = useRef(null)
    const socketRef = useRef(null); 
    const captureIntervalRef = useRef(null);
    const drawIntervalRef = useRef(null);
    const drawingRef = useRef(null);
    const sendingRef = useRef(null);
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
                ctx.strokeRect(edge[0], edge[1], edge[2] - edge[0], edge[3] - edge[1]);
            });
        }
    };

    const sendImage = () => {
        const video = videoRef.current;
        const canvas = canvasRefSending.current;

        const canvasW = canvas.width;
        const canvasH = canvas.height;

        const ctx = ctxRefSending.current;
        if (!ctx) return;

        ctx.drawImage(video, 0, 0, canvasW, canvasH);

        canvas.toBlob(async (blob) => {
            if (!blob) return;

            setLoading(true)

            try {
                socketRef.current?.send(blob)
            } catch (e) {
                setError(true);
            }
            finally {
                setLoading(false);
            }
        }, "image/jpeg", 0.9);
    };

    const printDefault = () => {
        const ctx = ctxRefBBox.current;
        if (!ctx) return;

        const img = new Image();
        img.src = "/cherry_bird.jpeg";

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

            if (drawingRef.current) {
                clearInterval(drawingRef.current);
                drawingRef.current = null;
            }

            if (sendingRef.current) {
                clearInterval(sendingRef.current);
                sendingRef.current = null;
            }

            bboxes.current = null;

            printDefault();
            setRunning(false);
            return;
        }

        const socket = new WebSocket("wss://api.marvinkrausser.com/predict_face");
        socketRef.current = socket;

        socket.onopen = () => {
            console.log("Connected");
            sendingRef.current = setInterval(sendImage, 100);
        }
        socket.onmessage = (event) => {
            drawBBox();
            bboxes.current = JSON.parse(event.data).bboxes;
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
                    Ask Expert
                </label>
            </div>
            <div id={styles.container}>
                <video autoPlay={true} ref={videoRef} style={{ width: "100%", height: "100%", position: "absolute", zIndex: "3" }}></video>
                <canvas ref={canvasRefBBox} style={{width: "100%", height: "100%", position: "absolute", zIndex: "4"}} />
                <canvas ref={canvasRefSending} style={{width: "100%", height: "100%", display: "none"}} />
            </div>
        </div>
    )
}

export default Object_Detection;