import { Link } from 'react-router-dom';
import './Menu.css'

function Menu() {
    return (
        <>
            <nav id="navbar-main">
                <Link className='navbar-item' to="/">Homepage</Link>
                <Link className='navbar-item' to="/bird_cnn">Birds</Link>
            </nav>
        </>
    );
}

export default Menu;