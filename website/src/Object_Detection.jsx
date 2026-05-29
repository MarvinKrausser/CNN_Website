import styles from './Object_Detection.module.css';
import { useRef, useEffect, useState } from 'react'

function Object_Detection() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [error, setError] = useState(false);
    const [loading, setLoading] = useState(false);
    const [bboxes, setBboxes] = useState(null)

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

    const captureImage = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;

        const width = video.videoWidth;
        const height = video.videoHeight;

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d");

        ctx.drawImage(video, 0, 0, width, height);

        ctx.strokeStyle = "red";
        ctx.lineWidth = 4;

        if (bboxes) {
            bboxes.forEach(element => {
                ctx.strokeRect(element[0], element[0], element[0], element[0]);
            });
        }

        canvas.toBlob(async (blob) => {
            if (!blob) return;

            setLoading(true)

            try {
                socket.send(file);
            } catch (e) {
                setError(true);
            }
            finally {
                setLoading(false);
            }
        }, "image/jpeg", 0.9);
    };

    useEffect(() => {
        const socket = new WebSocket("wss://api.marvinkrausser.com/predict_face");

        socket.onopen = async () => {
            console.log("Connected");
            const response = await fetch("/cherry_bird.jpeg");
            const blob = await response.blob();

            socket.send(blob);
        };

        socket.onmessage = (event) => {
            console.log("message")
            console.log(event.data);
        };

        socket.onerror = (err) => {
            console.error("WebSocket error:", err);
        };

        socket.onclose = () => {
            console.log("Disconnected");
        };

        // cleanup when component unmounts
        return () => {
            socket.close();
        };
    }, []);

    return (
        <div className='site-box'>
            <h1 className='site-headline'>Face Detection</h1>
            <div id={styles.container}>
                <video autoPlay={true} ref={videoRef} style={{ display: "none" }}></video>
                <canvas ref={canvasRef} width={640} height={480} />
            </div>
            <button onClick={captureImage}>Click</button>
        </div>
    )
}

export default Object_Detection;