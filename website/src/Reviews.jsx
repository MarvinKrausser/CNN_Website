import { useState, useRef, useEffect } from 'react';
import './Reviews.css';
import SelectDropdown from './components/SelectDropdown';

function Reviews() {
    const [error, setError] = useState(null);
    const [selectedWebsite, setSelectedWebsite] = useState("Not Specified");
    const [selectedRating, setSelectedRating] = useState(1);
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
        console.log(selectedWebsite);
    };

    const changeSelectedRating = (rating) => {
        setSelectedRating(rating);
        console.log(selectedRating);
    }

    const websites = [
        "CNN (current)",
        "Portfolio",
        "Not Specified"
    ];

    const ratings = [1, 2, 3, 4, 5];

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
                <div className='input-component-box'>
                    <p className='description'>Website</p>
                    <SelectDropdown
                        listClass="flex"
                        parentClass="input-component"
                        values={websites}
                        defaultValue={websites.at(-1)}
                        classNames={"website"}
                        changeInputParent={changeSelectedWebsite}
                        flex={true}
                    />
                </div>

                <div className='input-component-box'>
                    <p className='description'>Description</p>
                    <textarea
                        autoCorrect="off"
                        autoCapitalize="off"
                        spellCheck={false}
                        className='review-text input-component'
                        type="text"
                        value={text}
                        onChange={(e) => handleTextInput(e)}
                        placeholder="Type something..."
                    />
                    <p className='description second'>{text.length}/{MAX_TEXT}</p>
                </div>

                <div className='input-component-box'>
                    <p className='description'>Rating</p>
                    <SelectDropdown
                        listClass="grid"
                        parentClass="input-component"
                        values={ratings}
                        defaultValue={ratings.at(0)}
                        classNames={"rating"}
                        changeInputParent={changeSelectedRating}
                        flex={false}
                    />
                </div>
            </div>
        </div>
    </>)
}

export default Reviews;