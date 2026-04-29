import { Link } from 'react-router-dom';
import './Menu.css'

function Menu() {
    return (
        <>
            <nav id="navbar-main">
                <Link to="/">Homepage</Link>
                <Link to="/bird_cnn">Birds</Link>
            </nav>
        </>
    );
}

export default Menu;