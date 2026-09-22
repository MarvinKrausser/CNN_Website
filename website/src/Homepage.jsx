import { Link } from 'react-router-dom';
import styles from './Homepage.module.css';

const projects = [
    {
        to: "/bird_cnn",
        title: "Bird Species Expert",
        description: "A convolutional neural network that classifies bird species from a photo, with a live confidence score.",
        tag: "CNN · PyTorch",
    },
    {
        to: "/object_detection",
        title: "Face Detection",
        description: "A YOLO object detection model running on a live webcam feed, streamed to the server over a websocket.",
        tag: "YOLO · Realtime",
    },
    {
        to: "/reviews",
        title: "Reviews",
        description: "Leave a short, anonymous review of this site or my portfolio.",
        tag: "Feedback",
    },
];

function Homepage() {
    return (
        <div className='site-box'>
            <h1 className='site-headline'>Marvin Krausser</h1>

            <p className={styles.intro}>
                I build and deploy small machine learning projects end to end &mdash;
                from training the model to serving it behind a live API. Below are a
                couple of interactive demos, running on real models.
            </p>

            <div className={styles.grid}>
                {projects.map(({ to, title, description, tag }) => (
                    <Link key={to} to={to} className={`card ${styles.tile}`}>
                        <span className={styles.tag}>{tag}</span>
                        <h2 className={styles.tileTitle}>{title}</h2>
                        <p className={styles.tileText}>{description}</p>
                        <span className={styles.cta}>Try it &rarr;</span>
                    </Link>
                ))}
            </div>
        </div>
    );
}

export default Homepage;
