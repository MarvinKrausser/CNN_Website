import { useState, useEffect } from 'react';

function DropwdownFlex(props) {
    const values = props.values;
    const classNames = props.classNames;
    const changeInputParent = props.changeInputParent;
    const selectedInput = props.selectedInput;

    return (
        <div className={`review-select-menu flex ${classNames}`}>
            {values.map((item, i) =>
                String(selectedInput) !== String(item) && (
                    <div
                        key={item}
                        className={i >= (values.length - (selectedInput == values.at(-1) ? 2 : 1)) ? `review-select-item ${classNames} last` : `review-select-item ${classNames}`}
                        onClick={() => changeInputParent(item)}
                    >
                        {item}
                    </div>
                )
            )}
        </div>
    );
}

export default DropwdownFlex;