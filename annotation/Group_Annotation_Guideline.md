# Guideline 2: Component Grouping Guide

## 1. Introduction

After annotating components across multiple pages of a website, the next step is to group structurally and visually similar components together. This guide explains how to identify similarities and create meaningful component groups.

## 2. Purpose of Grouping

Grouping similar components helps:
- Identify reusable design patterns across the website
- Understand design consistency and variations
- Analyze component usage patterns
- Create a component taxonomy for the website

## 3. Definition of a Component Group

A **component group** is a collection of annotated components that share:
- **Similar structure**: Same type of elements arranged in comparable ways
- **Similar visual appearance**: Consistent styling, layout, and design patterns
- **Same functional purpose**: Serving the same role in the user interface
- **Comparable complexity**: Similar level of detail and nested elements

## 4. Grouping Criteria

Components should be grouped together if they satisfy **ALL** of the following criteria:

### 4.1 Structural Similarity (REQUIRED)
Components must have the same fundamental structure.

**Examples of structural similarity:**
- Both have an image on the left and text on the right
- Both contain: heading + description + button
- Both are horizontal lists of links
- Both have a grid layout with 3 columns

**NOT structurally similar:**
- One has image + text, another has only text
- One is horizontal, another is vertical
- One has 2 columns, another has 3 columns

### 4.2 Visual Similarity (REQUIRED)
Components must look similar in terms of design and styling.

**Visual attributes to compare:**
- **Layout**: Spacing, alignment, padding, margins
- **Typography**: Font families, sizes, weights (similar, not necessarily identical)
- **Color scheme**: Similar color palette or usage patterns
- **Visual hierarchy**: Similar emphasis and organization
- **Border/shadow**: Similar use of borders, shadows, or backgrounds
- **Iconography**: Similar icon style and placement

**Acceptable variations within a group:**
- Different text content
- Different images/media content
- Minor color variations (e.g., hover states, brand colors)
- Small spacing adjustments for responsive layouts

**NOT acceptable variations:**
- Completely different color schemes (dark vs light theme)
- Different typography systems (serif vs sans-serif)
- Major layout differences (card vs list view)

### 4.3 Functional Equivalence (REQUIRED)
Components must serve the same purpose or role.

**Examples of functional equivalence:**
- All are navigation menus (even if placed in different locations)
- All are product cards displaying items for sale
- All are call-to-action buttons with primary emphasis
- All are form input fields for text entry

**NOT functionally equivalent:**
- Navigation menu vs breadcrumb (both for navigation but different purposes)
- Primary button vs text link (both clickable but different emphasis)
- Product card vs blog post card (different content types)

### 4.4 Complexity Similarity (RECOMMENDED)
Components should have similar levels of detail.

**Similar complexity:**
- Both cards have: image + title + price + button (4 elements)
- Both forms have: 3 input fields + 1 submit button

**Different complexity:**
- One card has image + title, another has image + title + price + description + button
- One form has 2 fields, another has 10 fields with validation messages

**Note**: Some complexity variation is acceptable if structure and function align.

## 5. Grouping Process

### Step 1: Initial Sorting
Sort all annotated components by their type (buttons, cards, forms, navigation, etc.).

### Step 2: Visual Clustering
Within each type, visually compare components and create preliminary clusters based on appearance.

### Step 3: Structural Verification
For each cluster, verify that components share the same structural elements and layout.

### Step 4: Functional Validation
Confirm that all components in a group serve the same functional purpose.

### Step 5: Refinement
Review each group and split or merge as needed:
- **Split** if there are distinct sub-patterns
- **Merge** if differences are only in content, not structure/style

### Step 6: Group Naming
Assign a descriptive name to each group:
```
[ComponentType]_[DesignPattern]_[Location/Context]

Examples:
- Card_Product_Grid
- Button_Primary_CTA
- Navigation_Horizontal_Main
- Form_Contact_Sidebar
- Card_BlogPost_Featured
```

## 6. Grouping Rules & Decision Trees

### Rule 1: Same Component Type
✅ Components must be the same fundamental type to be grouped together.

**Example:**
- ✅ All buttons can potentially be in button groups
- ❌ Don't mix buttons with links, even if they look similar

### Rule 2: Structural Template Match
✅ Components must follow the same structural template.

**Decision Tree:**
```
Do components have the same elements?
├─ YES: Do elements appear in same order?
│   ├─ YES: Structure matches → Continue to next rule
│   └─ NO: Different structure → Separate groups
└─ NO: Different structure → Separate groups
```

### Rule 3: Visual Style Consistency
✅ Components should use consistent visual styling.

**Decision Tree:**
```
Do components have similar visual treatment?
├─ YES: Are layout and spacing patterns similar?
│   ├─ YES: Visual style matches → Continue to next rule
│   └─ NO: Different visual patterns → Consider separate groups
└─ NO: Different visual styles → Separate groups
```

**Exception**: Hover states, active states, and disabled states of the same component type can be in the same group.

### Rule 4: Cross-Page Consistency
✅ Components from different pages can be in the same group if they meet all criteria.

**Example:**
- Product cards on homepage and product listing page → Same group (if matching criteria)
- Navigation on homepage vs. navigation on article page → Same group (if matching criteria)

### Rule 5: Responsive Variations
✅ Desktop and mobile versions of the same component belong in the same group if they maintain structural and functional equivalence.

**Acceptable:**
- Desktop: horizontal navigation → Mobile: hamburger menu with same links
- Desktop: 4-column card grid → Mobile: single-column card stack

**Not acceptable:**
- Desktop: card with image → Mobile: list item without image (structural change)

## 7. Special Cases & Edge Cases

### 7.1 Component Variations
Some components have intentional variations (primary vs. secondary buttons, large vs. small cards).

**Approach:**
- **Option A**: Create separate groups for each variation if styling is significantly different
  - Group: Button_Primary
  - Group: Button_Secondary
  - Group: Button_Text

- **Option B**: Use a single group with sub-categories if variations are minor
  - Group: Button_Standard (includes primary, secondary, tertiary as variations)

**Guideline**: If variations are standard design system choices, use Option A. If they're contextual adjustments, consider Option B.

### 7.2 Hybrid Components
Some components share characteristics with multiple types.

**Example**: A card that also functions as a button (entire card is clickable)

**Approach**:
- Classify based on the **dominant characteristic**
- In this case: Card_Clickable (not Button)
- Note the hybrid nature in group description

### 7.3 Partial Matches
Components that match 2 out of 3 criteria.

**Approach**:
- If structure + function match but visual differs → Separate groups
- If structure + visual match but function differs → Separate groups
- If visual + function match but structure differs → Separate groups

**Rationale**: All three criteria must be met for grouping.

### 7.4 Progressive Enhancement
Components that look similar but have different levels of interactivity.

**Example**: Static card vs. card with hover effects vs. card with animations

**Approach**: Group together if the base structure and appearance are the same. Enhanced interactions are acceptable variations.

### 7.5 Semantic vs. Presentational Similarity
Sometimes components look similar but serve different semantic purposes.

**Example**: 
- A hero section banner
- A promotional banner
- Both have image + text + button but different semantic meaning

**Approach**: Consider context and purpose. If they're used interchangeably in the design system, group them. If they have distinct purposes, separate them.

## 8. Examples of Component Groups

### Example 1: Product Card Group

**Group Name**: Card_Product_Standard

**Included Components:**
- All product cards from homepage (6 instances)
- All product cards from category pages (multiple pages, 20+ instances)
- Featured product cards from landing pages (3 instances)

**Shared Characteristics:**
- Structure: Product image (top) + Product name + Price + "Add to Cart" button
- Visual: White background, rounded corners, drop shadow on hover
- Function: Display product information and enable purchase action
- Complexity: 4 main elements with consistent hierarchy

**Excluded from this group:**
- Product cards on comparison page (different layout: side-by-side specs)
- Mini product cards in checkout (simpler structure: image + name only)

### Example 2: Navigation Group

**Group Name**: Navigation_Horizontal_Main

**Included Components:**
- Main navigation on homepage
- Main navigation on product pages
- Main navigation on about page

**Shared Characteristics:**
- Structure: Horizontal list of text links with dropdowns
- Visual: Dark background, white text, underline on hover
- Function: Primary site navigation
- Complexity: 5-7 top-level items with nested submenus

**Excluded from this group:**
- Footer navigation (different structure: multi-column layout)
- Mobile hamburger menu (different visual presentation)
- Sidebar navigation (vertical orientation)

### Example 3: Button Group (with variations)

**Group Name**: Button_Primary_Action

**Included Components:**
- "Add to Cart" buttons on product cards
- "Submit" button on contact form
- "Get Started" button on hero sections
- "Download Now" buttons on landing pages

**Shared Characteristics:**
- Structure: Text label (optionally with icon)
- Visual: Solid background color (brand primary), white text, rounded corners
- Function: Primary call-to-action
- Complexity: Single-level, high-emphasis interactive element

**Acceptable variations within group:**
- Different text labels
- With or without icon
- Slightly different sizes (small, medium, large)
- Hover/active states

**Excluded from this group:**
- Secondary buttons (outlined style, not solid)
- Text links (no button appearance)
- Icon-only buttons (different structure)

## 9. Quality Assurance Checklist

For each component group, verify:

- [ ] All components in the group are the same type
- [ ] All components share the same structural elements
- [ ] Visual styling is consistent across components
- [ ] All components serve the same functional purpose
- [ ] Complexity levels are comparable
- [ ] Group name is descriptive and follows naming convention
- [ ] Edge cases are handled according to guidelines
- [ ] Cross-page instances are included appropriately
- [ ] Variations are acceptable within the group criteria

## 10. Group Documentation Template

For each group, document:

```markdown
## Group ID: [Unique identifier]

**Group Name**: [Descriptive name]

**Component Type**: [Button/Card/Form/Navigation/etc.]

**Total Instances**: [Number of components in this group]

**Pages Found**: [List of pages where components appear]

**Structural Pattern**:
- Element 1: [Description]
- Element 2: [Description]
- Element 3: [Description]
[...]

**Visual Characteristics**:
- Layout: [Description]
- Colors: [Description]
- Typography: [Description]
- Spacing: [Description]

**Functional Purpose**: [What this component does]

**Acceptable Variations**: [What can differ within this group]

**Example Instances**: [Reference 2-3 specific component IDs]

**Notes**: [Any special considerations or edge cases]
```

## 11. Handling Disagreements

When annotators disagree on grouping:

### Resolution Process:
1. **Review the criteria**: Check which of the 3 main criteria (structure, visual, function) is in question
2. **Examine specific instances**: Compare the disputed components side-by-side
3. **Consult examples**: Refer to similar cases in this guideline
4. **Document the disagreement**: Note the specific points of contention
5. **Team discussion**: If unresolved, bring to team meeting for consensus
6. **Establish precedent**: Document the decision for similar future cases

### Common Disagreement Scenarios:
- **Minor visual differences**: If 80%+ of visual attributes match, lean toward grouping together
- **Structural ambiguity**: If elements can be reordered without changing meaning, consider it the same structure
- **Functional overlap**: If components serve very similar but not identical purposes, create separate groups

## 12. Metrics for Quality Control

Track these metrics to ensure grouping quality:

- **Group size distribution**: Most groups should have 3+ instances
  - Single-instance groups may indicate over-splitting
  - 50+ instance groups may indicate under-splitting

- **Cross-page coverage**: Components should appear on multiple pages
  - If most groups are page-specific, reconsider grouping criteria

- **Consistency score**: Inter-annotator agreement rate
  - Aim for 80%+ agreement on group assignments
  - Lower rates indicate need for guideline clarification

## 13. Common Mistakes to Avoid

❌ **Over-splitting**: Creating too many groups for minor variations
❌ **Under-splitting**: Grouping dissimilar components because they're the same type
❌ **Content-based grouping**: Grouping by content rather than structure/appearance
❌ **Page-specific bias**: Not recognizing cross-page patterns
❌ **Ignoring function**: Grouping visually similar but functionally different components
❌ **Being too rigid**: Not allowing acceptable variations within groups
❌ **Inconsistent application**: Using different criteria for different component types

## 14. Review and Iteration

After initial grouping:

1. **Cross-check with other annotators**: Compare grouping decisions
2. **Review outliers**: Examine single-instance groups and very large groups
3. **Test consistency**: Can another annotator assign a new component to existing groups?
4. **Refine as needed**: Adjust group boundaries based on review findings
5. **Update documentation**: Record any clarifications or special cases discovered

## 15. Deliverables

For each website, provide:

1. **Group Registry**: List of all component groups with IDs and descriptions
2. **Component-to-Group Mapping**: Each annotated component assigned to a group
3. **Group Examples**: Visual examples of 2-3 components from each group
4. **Grouping Rationale**: Brief explanation for ambiguous grouping decisions
5. **Statistics Report**: Group counts, distribution, coverage metrics

---

**Remember**: The goal is to identify meaningful design patterns that reflect how components are actually used across the website. When in doubt, prioritize consistency in structure and function over minor visual differences.