# Guideline 1: Web Component Annotation Guide

## 1. Introduction

This guide provides instructions for annotating UI components across multi-page websites. Your goal is to identify and mark discrete, functional UI elements that appear on web pages.

## 2. Definition of a Component

A **component** is a self-contained, visually distinct UI element that serves a specific function or purpose on a webpage. Components have:

- **Clear boundaries**: Distinguishable visual edges or spacing from surrounding elements
- **Functional purpose**: A specific role in user interaction or information display
- **Structural integrity**: Complete within itself (not a fragment of a larger element)
- **Reusability potential**: Could theoretically be reused elsewhere on the site

## 3. Component Categories & Examples

### 3.1 Navigation Components
Elements that help users move through the website.

**Examples:**
- **Navigation Bar**: Horizontal or vertical menu with links to main sections
- **Breadcrumb**: Path showing current location (e.g., Home > Products > Laptops)
- **Sidebar Menu**: Vertical navigation list, often collapsible
- **Pagination**: Controls for navigating through pages (Previous, 1, 2, 3, Next)
- **Tab Navigation**: Horizontal tabs switching between content sections

### 3.2 Content Display Components
Elements that present information to users.

**Examples:**
- **Hero Section**: Large banner with headline, subtext, and call-to-action button
- **Product Card**: Box containing product image, name, price, and action button
- **Article Card**: Preview of blog post with thumbnail, title, excerpt, and metadata
- **Testimonial**: Customer quote with name, photo, and rating
- **Stat Counter**: Numerical display with icon and label (e.g., "500+ Customers")
- **Feature List Item**: Icon + heading + description in a features section
- **Accordion Item**: Collapsible content section with header

### 3.3 Interactive Components
Elements that enable user actions.

**Examples:**
- **Button**: Single clickable element (primary, secondary, or text button)
- **Form**: Complete input structure with fields and submit button
- **Search Bar**: Input field with search icon/button
- **Dropdown Menu**: Selector with multiple options
- **Modal/Dialog**: Overlay window with content and close button
- **Slider/Carousel**: Rotating content display with navigation arrows
- **Toggle Switch**: On/off control element
- **Checkbox/Radio Group**: Set of related selection inputs

### 3.4 Informational Components
Elements that provide static or dynamic information.

**Examples:**
- **Footer**: Bottom section with links, contact info, social media, copyright
- **Header**: Top section typically with logo, navigation, and utilities
- **Alert/Banner**: Notification bar (success, warning, error, info)
- **Badge**: Small label indicating status (e.g., "New", "Sale", "Beta")
- **Tooltip**: Small popup with additional information on hover
- **Progress Bar**: Visual indicator of completion or loading status
- **Table**: Structured data display with rows and columns
- **Chart/Graph**: Data visualization element

### 3.5 Media Components
Elements that display visual or multimedia content.

**Examples:**
- **Image Gallery**: Collection of images with thumbnails
- **Video Player**: Embedded video with controls
- **Icon**: Single graphical symbol (when standalone and functional)
- **Logo**: Brand identity graphic
- **Avatar**: User profile picture or placeholder

## 4. Component Boundaries: What to Include

### 4.1 Complete Functional Units
✅ **DO annotate** the entire functional unit:
- A product card including image, title, price, and "Add to Cart" button
- A navigation bar with all its menu items
- A form with all input fields, labels, and submit button

❌ **DON'T annotate** individual parts separately:
- Just the image from a product card
- Individual menu items from a navigation bar
- Single input fields from a form (unless they function independently)

### 4.2 Nested Components - IMPORTANT RULE
**Only annotate the outermost component when components are nested.**

When a component contains other components within it, annotate ONLY the parent/container component. Do NOT create separate annotations for child components.

**Rationale**: This approach simplifies code generation, as the entire component will be implemented as a single block with its internal structure intact.

**Example 1:** A card containing a button
- ✅ Annotate the entire card as one component (including the button inside)
- ❌ Do NOT annotate the button separately

**Example 2:** A header containing a navigation bar and logo
- ✅ Annotate the entire header as one component (including nav and logo)
- ❌ Do NOT annotate the navigation bar separately
- ❌ Do NOT annotate the logo separately

**Example 3:** A form containing multiple input fields and a submit button
- ✅ Annotate the entire form as one component
- ❌ Do NOT annotate individual input fields or the submit button

**Exception**: Only annotate a nested element separately if it appears BOTH as a standalone component elsewhere on the site AND as a nested element. In this case, annotate both instances.

### 4.3 Repeated Elements - CRITICAL FOR CODE GENERATION
**When similar elements repeat consecutively, annotate them as a SINGLE GROUP annotation, not individual components.**

This is essential for efficient code implementation using loops/iteration.

**What counts as "consecutive repetition":**
- Multiple product cards in a grid or list
- A series of feature items in a features section
- Multiple testimonial cards in a row
- List items in a navigation menu
- Table rows with similar structure
- Gallery images in sequence

**How to annotate repeated elements:**

✅ **CORRECT - Single annotation for the repeating pattern:**
```
Component: CardGroup_Product_1
- Instances: 12 product cards
- Pattern: Repeating grid of identical card structure
- Boundary: Entire grid/list of cards
```

❌ **INCORRECT - Individual annotations:**
```
Component: Card_Product_1
Component: Card_Product_2
Component: Card_Product_3
... (Don't do this!)
```

**Implementation benefit:**
- Single annotation → Generates code with `for` loop or `.map()` function
- Individual annotations → Generates repetitive code for each instance

**When to annotate separately:**
Even if components look similar, annotate them separately if:
- They are NOT consecutive (e.g., separated by other different components)
- They appear in different sections of the page
- They have significantly different content or structure

**Example scenarios:**

**Scenario 1: Product grid ✅ Annotate as one group**
```
[Product Card] [Product Card] [Product Card]
[Product Card] [Product Card] [Product Card]
```
→ Single annotation: CardGroup_Product_Grid_1 (6 instances)

**Scenario 2: Separated cards ❌ Annotate separately**
```
[Product Card]
[Banner Advertisement]
[Product Card]
[Text Section]
[Product Card]
```
→ Three separate annotations (not consecutive)

**Scenario 3: Different sections ❌ Annotate separately**
```
Featured Products Section:
  [Product Card] [Product Card]

...other content...

Related Products Section:
  [Product Card] [Product Card]
```
→ Two group annotations (different semantic sections)

## 5. Edge Cases & Special Scenarios

### 5.1 Text Blocks
- **Plain paragraphs**: Generally NOT components unless they have distinct styling/function
- **Styled text blocks**: YES, if they have special formatting (callout boxes, quotes)
- **Headings**: NOT components by themselves, but part of the section they introduce

### 5.2 Whitespace and Containers
- **Empty containers**: If they have visible borders/background, annotate them
- **Layout grids**: Do NOT annotate grid structures themselves, only the content within

### 5.3 Decorative vs Functional
- **Decorative images**: NOT components (e.g., background patterns)
- **Functional images**: YES, components (e.g., product photos, logos, icons with actions)
- **Divider lines**: NOT components unless they're part of a larger element

### 5.4 Overlapping Elements
If elements visually overlap:
- Annotate based on the DOM structure when visible
- Prioritize the foreground element
- Note: Both elements should be annotated if both are functional

## 6. Annotation Process

### Step 1: Scan the Page
Review the entire page to understand its structure and identify major sections.

### Step 2: Identify Components
Look for visually distinct, functional elements as defined in Section 3.

### Step 3: Mark Boundaries
Draw a bounding box around each component that:
- Includes all visual elements of the component
- Includes necessary padding/margins that are part of the design
- Excludes surrounding whitespace that's not part of the component
- **For repeated elements**: Encompasses the entire repeating group, not individual instances
- **For nested elements**: Only marks the outermost component boundary

### Step 4: Label Components
Assign a descriptive label following this format:

**For single components:**
```
[ComponentType]_[Location/Purpose]_[Index]

Examples:
- Button_AddToCart_1
- Card_Product_3
- Navigation_Main_1
- Form_ContactUs_1
```

**For repeating component groups:**
```
[ComponentType]Group_[Location/Purpose]_[Index]
Include instance count in the annotation

Examples:
- CardGroup_Product_Grid_1 (12 instances)
- ListItemGroup_Features_1 (6 instances)
- TestimonialGroup_HomePage_1 (4 instances)
```

### Step 5: Document Component Details
For each annotation, document:

**For single components:**
- Component type
- Functional purpose
- Location on page

**For repeating component groups:**
- Component type
- Number of instances in the group
- Repeating pattern description
- Functional purpose
- Location on page

**Note**: With the new nested component rule, you generally won't need to document parent-child relationships since only the outermost component is annotated.

## 7. Quality Checklist

Before submitting your annotations, verify:

- [ ] All functional UI elements are annotated
- [ ] Component boundaries are accurate and complete
- [ ] Labels are descriptive and follow naming conventions
- [ ] Nested components are properly documented
- [ ] No duplicate annotations for the same element
- [ ] Consistent annotation style across all pages
- [ ] Edge cases are handled according to guidelines

## 8. Common Mistakes to Avoid

❌ **Too granular**: Annotating individual words or tiny decorative elements
❌ **Too broad**: Marking entire page sections as single components (unless they are cohesive functional units)
❌ **Inconsistent boundaries**: Including padding on some components but not others
❌ **Annotating nested components separately**: Remember to only annotate the outermost component
❌ **Separating repeated elements**: Consecutive repeated components should be grouped in a single annotation
❌ **Misclassifying**: Labeling a card as a button or vice versa

## 9. Examples of Complete Annotations

### Example 1: E-commerce Product Listing Page

**Components to annotate:**
1. Header_Main_1 (entire top bar including logo, navigation, search bar, and cart button)
2. Breadcrumb_1 (Home > Electronics > Phones)
3. Filter_Sidebar_1 (left sidebar with all filter options)
4. **CardGroup_Product_Grid_1 (12 instances)** - All product cards in the main grid
5. Pagination_1 (bottom navigation)
6. Footer_Main_1 (bottom section with all links and info)

**Note**: Unlike previous approach, we now:
- Group all 12 product cards into ONE annotation (not 12 separate ones)
- Don't separately annotate the buttons, images inside the cards
- Don't separately annotate elements inside the header

### Example 2: Blog Article Page

**Components to annotate:**
1. Header_Main_1 (complete header with all nested elements)
2. Hero_Article_1 (featured image + headline + metadata combined)
3. ShareButtons_Social_1 (all sharing buttons as one component)
4. Alert_Newsletter_1 (subscription banner)
5. **CardGroup_RelatedArticles_1 (3 instances)** - All related article cards together
6. Form_Comment_1 (complete comment form with all fields and submit button)
7. Footer_Main_1 (complete footer)

### Example 3: Homepage with Features Section

**Components to annotate:**
1. Header_Main_1
2. Hero_Homepage_1 (main hero section)
3. **FeatureItemGroup_Benefits_1 (6 instances)** - Grid of 6 feature items (icon + heading + description)
4. CTA_Newsletter_1 (call-to-action section)
5. **TestimonialCardGroup_Customer_1 (4 instances)** - Row of testimonial cards
6. Footer_Main_1

**Implementation benefit:**
```javascript
// With grouped annotation - clean loop-based code
features.map(feature => (
  <FeatureItem 
    icon={feature.icon}
    heading={feature.heading}
    description={feature.description}
  />
))

// vs. old approach - repetitive hardcoded elements
<FeatureItem icon="..." heading="..." description="..." />
<FeatureItem icon="..." heading="..." description="..." />
<FeatureItem icon="..." heading="..." description="..." />
// ...repeated 6 times
```

## 10. Getting Help

If you encounter ambiguous cases:
1. Document the element with a screenshot
2. Note your question/concern
3. Flag for team discussion
4. Use your best judgment in the meantime based on these guidelines

## 11. Tools and Interface Notes

[Add specific instructions for your annotation tool here]
- How to create bounding boxes
- How to assign labels
- How to mark nested relationships
- How to save and submit annotations

---

**Remember**: Consistency is key. When in doubt, ask yourself: "Is this a self-contained, functional UI element that serves a clear purpose?" If yes, it's likely a component.