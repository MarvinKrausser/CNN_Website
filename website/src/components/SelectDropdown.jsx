import { useState, useRef, useEffect } from 'react';

function SelectDropdown(props) {
    const values = props.values;
    const defaultValue = props.defaultValue;
    const classNames = props.classNames;
    const changeInputParent = props.changeInputParent;
    const [selectedInput, setselectedInput,] = useState(defaultValue);
    const [selectVisibility, setSelectVisibility] = useState(false);
    const SelectButton = useRef();

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
        setselectedInput(input);
        changeInputParent(input);
    };


    return (
        <div className={`review-select-dropdown ${classNames}`} >
            <button ref={SelectButton} className={selectVisibility ? `review-select-button ${classNames} active` : `review-select-button ${classNames}`} onClick={() => { setSelectVisibility(!selectVisibility) }}>
                {selectedInput}
            </button>
            {selectVisibility && <div className={`review-select-menu ${classNames}`}>
                {values.map((item, i) =>
                    String(selectedInput) !== String(item) && (
                        <div
                            key={item}
                            className={i >= (values.length - (selectedInput == values.at(-1) ? 2 : 1)) ? `review-select-item ${classNames} last` : `review-select-item ${classNames}`}
                            onClick={() => changeSelectedInput(item)}
                        >
                            {item}
                        </div>
                    )
                )}
            </div>}
        </ div>
    );
}

export default SelectDropdown;