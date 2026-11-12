# Style Guide

## Colours

| Role                  | Hex       | Usage                                   |
| --------------------- | --------- | --------------------------------------- |
| Background            | `#260003` | Main page background                    |
| Secondary background  | `#6E0025` | Cards, panels, sections                 |
| Hover background      | `#8F0030` | (Cards, panels, sections) Hovered       |
| Font                  | `#FDDAE3` | Default text colour                     |
| Interactable          | `#B88A99` | Buttons, links, inputs                  |
| Hover                 | `#D6A1B3` | Hover state of interactable elements    |
| Not interactable      | `#635A5D` | Disabled or inactive elements           |

### CSS variables

```css
:root {
  --color-background:           #260003;
  --color-background-secondary: #6E0025;
  --color-background-hover:     #8F0030;
  --color-font:                 #FDDAE3;
  --color-interactable:         #B88A99;
  --color-hover:                #D6A1B3;
  --color-not-interactable:     #635A5D;
}
```


## Typography

### Font families

| Role      | Family                                  | Weights      | Usage                              |
| --------- | --------------------------------------- | ------------ | ---------------------------------- |
| Heading   | `'Redaction 50', system-ui, sans-serif` | 600, 700     | h1–h4, navbar brand                |
| Body      | `Roboto, -apple-system, sans-serif`     | 400, 700     | Default text, buttons              |
| Monospace | `'Space Mono', monospace`               | 400          | Tags, code/raw output, textareas   |


### Sizes

| Role      | Size                          | Line height | Usage                  |
| --------- | ----------------------------- | ----------- | ---------------------- |
| Headline  | `clamp(2.2rem, 8vw, 4rem)`    | 1.15        | Page title             |
| Body      | `1rem`                        | 1.7         | Default text           |
| Small     | `0.75rem`                     | 1.7         | Tags, labels           |