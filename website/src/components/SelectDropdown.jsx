import { useState, useRef, useEffect } from 'react';
import DropwdownFlex from './DropdownFlex';
import DropwdownGrid from './DropdownGrid';

function SelectDropdown(props) {
    const values = props.values;
    const defaultValue = props.defaultValue;
    const classNames = props.classNames;
    const changeInputParent = props.changeInputParent;
    const parentClass = props.parentClass;
    const flex = props.flex;
    const [selectVisibility, setSelectVisibility] = useState(false);
    const SelectButton = useRef();

    const [selectedInput, setSelectedInput] = useState(defaultValue);

    const SelectMenu =
        flex === true
            ? DropwdownFlex
            : DropwdownGrid;

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


    return (
        <div className={`review-select-dropdown ${classNames} ${parentClass}`} >
            <button ref={SelectButton} className={selectVisibility ? `review-select-button ${classNames} active` : `review-select-button ${classNames}`} onClick={() => { setSelectVisibility(!selectVisibility) }}>
                {selectedInput}
            </button>
            {selectVisibility &&
                <SelectMenu
                    classNames={classNames}
                    changeInputParent={changeSelectedInput}
                    values={values}
                    selectedInput={selectedInput}
                />}

        </ div>
    );
}

export default SelectDropdown;