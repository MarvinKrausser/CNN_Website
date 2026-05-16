import { useState, useRef, useEffect } from 'react';

function SelectDropdown(props) {
    const values = props.values;
    const defaultValue = props.defaultValue;
    const classNames = props.classNames;
    const changeInputParent = props.changeInputParent;
    const parentClass = props.parentClass;
    const styles = props.styles;
    const [selectVisibility, setSelectVisibility] = useState(false);
    const SelectButton = useRef();

    const [selectedInput, setSelectedInput] = useState(defaultValue);

    useEffect(() => {
        function handleClick(event) {
            if (SelectButton.current && !SelectButton.current.contains(event.target)
            ) {
                setSelectVisibility(false);
            }
        }

        document.addEventListener("click", handleClick);

        return () => {
            document.removeEventListener("click", handleClick);
        };
    }, []);

    const changeSelectedInput = (input) => {
        setSelectedInput(input);
        changeInputParent(input);
    };

    const CreateItems = () => {
        const indices = Array.from({ length: values.length - 1 }, (_, i) => values.length - 2 - i);
        return values.map((item, i) =>
            String(selectedInput) !== String(item) && (
                <div
                    key={item}
                    className={`${styles["review-select-item"]} ${styles[`review-select-box-${indices[indices.length - 1]}`]} ${classNames}` + (i >= (values.length - (selectedInput == values.at(-1) ? 2 : 1)) ? ` ${styles.last}` : '')}
                    onClick={() => changeSelectedInput(item)}
                    style={{gridArea:`box${indices.pop()}`}}
                >
                    {item}
                </div>
            )
        )
    }


    return (
        <div className={`${styles["review-select-dropdown"]} ${classNames} ${parentClass}`} >
            <button ref={SelectButton} className={selectVisibility ? `${styles["review-select-button"]} ${classNames} ${styles.active}` : `${styles["review-select-button"]} ${classNames}`} onClick={() => { setSelectVisibility(!selectVisibility) }}>
                {selectedInput}
            </button>
            {selectVisibility &&
                <div className={`${styles["review-select-menu"]} ${classNames}`} style={{display: "grid"}}>
                    <CreateItems />
                </div>}

        </ div>
    );
}

export default SelectDropdown;