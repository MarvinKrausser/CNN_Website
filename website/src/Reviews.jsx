import { useState, useRef, useEffect } from 'react';
import './Reviews.css';
import SelectDropdown from './components/SelectDropdown';

function Reviews() {
    const [error, setError] = useState(null);
    const [selectedWebsite, setSelectedWebsite] = useState("Not Specified");
    const [selectedRating, setSelectedRating] = useState(0);
    const [text, setText] = useState("");

    const MAX_TEXT = 180;


    const fetchReviews = async () => {
        const response = await fetch(`api.marvinkrausser/review`, {
            method: "GET",
        });

        if (!response.ok) {
            setError(true);
            return;
        }
        else {
            setError(false);
        }

        const result = await response.json();
    }

    const sendReview = async (review) => {

        const response = await fetch(`api.marvinkrausser/review`, {
            method: "POST",
            body: JSON.stringify(review),
        });

        if (!response.ok) {
            setError(true);
            return;
        }
        else {
            setError(false);
        }

        const result = await response.json();
    }

    const changeSelectedWebsite = (website) => {
        setSelectedWebsite(website);
    };

    const changeSelectedRating = (rating) => {
        setSelectedRating(rating);
    }

    const websites = [
        "CNN (current)",
        "Portfolio",
        "Not Specified"
    ];

    const ratings = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    const handleTextInput = (e) => {
        const textInput = e.target.value;

        if (textInput.length <= MAX_TEXT || textInput.length < text.length) {
            setText(textInput);
        }
    };


    return (<>
        <div className='site-box'>
            <h1 className='site-headline'>Reviews</h1>

            <div className='input-box'>
                <SelectDropdown values={websites} defaultValue={websites.at(0)} classNames={"website"} changeInputParent={changeSelectedWebsite} />

                <div className='review-text-box'>
                    <textarea
                        autoCorrect="off"
                        autoCapitalize="off"
                        spellCheck={false}
                        className='review-text'
                        type="text"
                        value={text}
                        onChange={(e) => handleTextInput(e)}
                        placeholder="Type something..."
                    />
                    <p>{text.length}/{MAX_TEXT}</p>
                </div>

                <SelectDropdown values={ratings} defaultValue={ratings.at(0)} classNames={"rating"} changeInputParent={changeSelectedRating} />
            </div>
        </div>
    </>)
}

export default Reviews;