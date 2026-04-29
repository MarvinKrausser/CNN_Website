import { useState } from 'react'
import { useRef } from "react";
import './Bird_CNN.css'

function Bird_CNN() {
    const apiUrl = process.env.NODE_ENV === "development"
        ? "http://134.60.154.7:8000"
        : "http://134.60.154.7:8000";

    const boxRef = useRef(null);

    const [file, setFile] = useState(null);
    const [birdClass, setBirdClass] = useState(null);
    const [confidence, setConfidence] = useState(null);
    const [error, setError] = useState(false);
    const [preview, setPreview] = useState(null);

    const handleImage = (e) => {
        setFile(e.target.files[0]);

        if (e.target.files[0]) {
            setPreview(URL.createObjectURL(e.target.files[0]));
            boxRef.current.classList.add("active");
        }
        setConfidence(null);
        setBirdClass(null);
    };

    const sendImage = async (e) => {
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(`${apiUrl}/predict`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                setError(true);
            }
            else {
                setError(false);
            }

            const result = await response.json();
            setBirdClass(result["class"]);
            const confidence = result["confidence"];
            setConfidence(`${Math.round(confidence * 100)}%`);
        }
        catch (error) {
            console.error("Upload failed:", error);
        }
    };

    return (
        <>
            <div id='background'></div>
            <div className='site-content'>
                <h1 id='site-headline'>Bird Species Expert</h1>
                <div className='input-box'>
                    <input type="file" id='fileUpload' accept="image/jpeg" onChange={handleImage} style={{ display: "none" }} />
                    <label htmlFor="fileUpload" className="custom-button">
                        Choose an image
                    </label>
                    <button id='button-send' onClick={sendImage} style={{ display: "none" }} />
                    <label htmlFor="button-send" className="custom-button">
                        Ask Expert
                    </label>

                </div>

                <div ref={boxRef} className='image-box'>
                    {preview && (
                        <img
                            src={preview}
                            alt="preview"
                            style={{ height: "300px", border: "1px solid black" }}
                        />
                    )}
                </div>


                <div className='response-block'>
                    {!error && <div className='content-block class'>
                        <h3>Bird Species: </h3>
                        <p id='bird-class-text'>{birdClass}</p>
                    </div>}
                    {!error && <div className='content-block confidence'>
                        <h4>Model Confidence: </h4>
                        <p id='bird-confidence-text'>{confidence}</p>
                    </div>}
                    {error && <h4>An Error has uccured. Please try again later.</h4>}
                </div>
            </div>
        </>
    );
}

export default Bird_CNN;