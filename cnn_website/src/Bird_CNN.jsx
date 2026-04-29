import { useState } from 'react'
import './Bird_CNN.css'

function Bird_CNN() {
    const apiUrl = process.env.NODE_ENV === "development"
        ? "http://127.0.0.1:8000"
        : "https://api.myapp.com";

    const [file, setFile] = useState(null);
    const [birdClass, setBirdClass] = useState(null);
    const [confidence, setConfidence] = useState(null);
    const [error, setError] = useState(false);

    const handleImage = (e) => {
        setFile(e.target.files[0]);
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
            <h1>bird-cnn</h1>
            <div>
                <input type="file" accept="image/jpeg" onChange={handleImage} />
                <button onClick={sendImage}>Upload</button>


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