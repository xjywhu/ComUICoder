# Guideline: Website and Subpage Selection

## 1. Introduction

This guide provides instructions for selecting websites from the top 500 most visited websites globally and identifying 2-5 important subpages within each selected website for component annotation. The goal is to build a diverse, high-quality dataset that represents real-world web design patterns.

## 2. Website Selection Source

Select websites from: **Moz Top 500 Global Websites**  
URL: https://moz.com/top500

This list represents the most visited and influential websites globally, ensuring that annotated components reflect industry-standard design patterns.

## 3. Website Selection Criteria

### 3.1 MUST Include: Design Complexity
✅ **Select websites that have:**
- Rich, multi-component layouts
- Diverse UI elements (navigation, cards, forms, buttons, etc.)
- Multiple distinct sections on each page
- Visual hierarchy and structured content organization
- Interactive elements beyond basic links

❌ **Do NOT select websites that have:**
- Extremely simple layouts (e.g., single-column text-only pages)
- Minimal UI components (less than 5 distinct component types per page)
- Primarily text-based content with little visual structure
- Search engine result pages (too homogeneous)

**Examples:**
- ✅ E-commerce sites (Amazon, eBay, Etsy) - rich product cards, filters, navigation
- ✅ News/media sites (CNN, BBC, Medium) - article cards, navigation, multimedia
- ✅ Social platforms (Twitter, LinkedIn, Reddit) - posts, profiles, feeds
- ❌ Wikipedia article pages - primarily text content, limited component variety
- ❌ Google search results - too simple and repetitive

### 3.2 MUST Include: Component Reusability
✅ **Select websites where components repeat across multiple pages:**
- Product cards appear on homepage, category pages, search results
- Navigation bars are consistent across different page types
- Footer sections are reused throughout the site
- Card layouts for different content types (articles, products, profiles)

❌ **Avoid websites where each page is completely unique:**
- Every page has different layout patterns
- No repeating component structures
- Inconsistent design systems

### 3.3 MUST Include: Multi-page Structure
✅ **Select websites with clear page hierarchies:**
- Homepage
- Category/listing pages
- Detail/individual item pages
- User account/profile pages
- About/information pages

❌ **Avoid:**
- Single-page applications where all content is on one page
- Websites with only 1-2 accessible page types

### 3.4 SHOULD Include: Design Quality
✅ **Prefer websites with:**
- Professional, modern design
- Clear visual hierarchy
- Consistent design system
- Well-structured HTML/CSS (inspect the code quality)
- Responsive design implementations

⚠️ **Be cautious with:**
- Outdated designs (likely represent deprecated patterns)
- Poorly structured code (difficult to annotate accurately)
- Heavy use of iframes or embedded content

### 3.5 MUST Include: Accessibility
✅ **Select websites that are:**
- Publicly accessible without login requirements (for at least some pages)
- Available in English or with clear visual structure regardless of language
- Not behind paywalls for the pages you intend to annotate
- Stable and unlikely to change dramatically during annotation period

❌ **Avoid:**
- Websites requiring authentication for all content
- Websites with heavy anti-scraping measures that prevent viewing
- Beta or frequently redesigned websites

### 3.6 SHOULD Include: Diversity
Across all selected websites, aim for diversity in:
- **Domain types**: E-commerce, news, social media, education, entertainment, tools/services
- **Design styles**: Minimalist, content-heavy, visual-first, data-driven
- **Geographic origin**: Western, Asian, European sites (different design conventions)
- **Target audience**: B2C, B2B, general public, professionals

### 3.7 MUST Exclude: Inappropriate Content
❌ **Never select websites with:**
- Adult content
- Gambling or betting sites
- Illegal or unethical services
- Hate speech or extremist content
- Excessive advertising that obscures content

### 3.8 Technical Considerations
✅ **Verify before selection:**
- Website loads properly in modern browsers
- Pages render correctly without JavaScript errors
- Content is visible and inspectable in browser DevTools
- No excessive pop-ups or interstitials that block content

## 4. Subpage Selection Criteria

After selecting a website, choose **2-5 subpages** that represent different page types and component usage patterns.

### 4.1 Required Subpage: Homepage
✅ **Always include the homepage** (unless it's unusable)
- Usually the most design-rich page
- Showcases the full range of components
- Sets the design system standards

### 4.2 Required Diversity: Different Page Types
Select subpages that represent **distinct page templates**:

**E-commerce websites (select 3-4):**
- Homepage
- Category/listing page (e.g., /products/electronics)
- Product detail page (e.g., /product/laptop-model-x)
- Search results page (optional)

**News/Media websites (select 3-4):**
- Homepage
- Section page (e.g., /world-news or /technology)
- Article detail page
- Author or topic archive page (optional)

**Social platforms (select 2-3):**
- Homepage/feed
- Profile page
- Individual post/thread page

**Service/Tool websites (select 2-3):**
- Homepage
- Features/products page
- Pricing page
- Documentation or help page (optional)

### 4.3 Component Coverage
Select subpages that collectively cover:
- **Navigation components**: Headers, menus, breadcrumbs, pagination
- **Content display**: Cards, lists, grids, articles, media
- **Interactive elements**: Forms, buttons, search bars, filters
- **Informational elements**: Footers, alerts, badges, metadata

**Goal**: The 2-5 subpages together should showcase at least 10-15 distinct component types.

### 4.4 Avoid Redundancy
❌ **Do NOT select subpages that are nearly identical:**
- Multiple product detail pages (choose just one representative)
- Multiple article pages with same template (choose just one)
- Multiple category pages with identical layouts

✅ **DO select subpages with different layouts:**
- Homepage (grid layout) + Category page (list layout) + Detail page (single-column)
- Feed page (cards) + Profile page (bio + grid) + Settings page (form)

### 4.5 Prioritize Important Pages
Select pages that are:
- **High-traffic pages**: Most visited sections of the website
- **Core functionality pages**: Essential to the website's purpose
- **Representative pages**: Typical examples of the website's design patterns

Avoid selecting:
- Rarely visited pages (e.g., terms of service, privacy policy)
- Error pages (404, 500)
- Login/signup pages (often simple and not representative)
- Empty state pages (e.g., empty cart, no search results)

## 5. Selection Process

### Step 1: Browse the Top 500 List
Review websites from the Moz Top 500 list systematically.

### Step 2: Quick Evaluation
For each candidate website, quickly assess:
- Can I access it without login? (check 2-3 pages)
- Does it have rich, diverse components?
- Are there multiple distinct page types?
- Is the design quality acceptable?

**Time limit**: Spend no more than 3-5 minutes per candidate website.

### Step 3: Deep Evaluation
For promising candidates:
- Navigate through 5-10 different pages
- Inspect HTML structure using DevTools
- Identify component patterns
- Count distinct component types
- Verify technical accessibility

**Time limit**: Spend 10-15 minutes per promising candidate.

### Step 4: Subpage Identification
Once a website is selected:
- Start with the homepage
- Navigate to identify different page templates
- Select 2-5 pages that maximize component diversity and pattern coverage
- Document the URL and page type for each subpage

### Step 5: Documentation
For each selected website, document:
```
Website: [Website Name]
URL: [Homepage URL]
Domain Type: [E-commerce/News/Social/Service/etc.]
Design Style: [Minimalist/Content-heavy/Visual-first/etc.]

Selected Subpages:
1. [Page Type]: [Full URL]
2. [Page Type]: [Full URL]
3. [Page Type]: [Full URL]
...

Rationale: [Brief explanation of why this website was selected]
Component Diversity: [Estimated number of distinct component types]
```

## 6. Quality Checklist

Before finalizing your selection, verify:

**Website Level:**
- [ ] Website is from Top 500 list
- [ ] Has sufficient design complexity (10+ component types visible)
- [ ] Components repeat across multiple pages
- [ ] Multiple page types are accessible
- [ ] No inappropriate content
- [ ] Website loads correctly in modern browsers
- [ ] Contributes to overall dataset diversity

**Subpage Level:**
- [ ] 2-5 subpages selected per website
- [ ] Homepage is included (unless unusable)
- [ ] Subpages represent different page templates
- [ ] Subpages collectively showcase diverse components
- [ ] No redundant/nearly identical subpages
- [ ] All subpages are publicly accessible
- [ ] URLs are documented correctly

## 7. Recommended Distribution

For a balanced dataset, aim for the following distribution across all selected websites:

**Domain Types:**
- 25-30%: E-commerce (Amazon, eBay, Shopify stores, etc.)
- 20-25%: News/Media (CNN, Medium, TechCrunch, etc.)
- 15-20%: Social Media/Community (Twitter, Reddit, LinkedIn, etc.)
- 15-20%: Technology/Software (GitHub, Microsoft, Adobe, etc.)
- 10-15%: Entertainment (Netflix, Spotify, YouTube, etc.)
- 5-10%: Education/Reference (Khan Academy, Coursera, etc.)

**Design Complexity:**
- 30-40%: High complexity (20+ component types)
- 40-50%: Medium complexity (10-19 component types)
- 10-20%: Moderate complexity (8-10 component types)
- 0%: Low complexity (fewer than 8 component types) - avoid

**Geographic Diversity:**
- Include websites with different regional design conventions
- Mix of Western, Asian, and European sites
- Different language interfaces (but ensure annotability)

## 8. Examples of Good Selections

### Example 1: Amazon.com
**Domain Type**: E-commerce  
**Why selected**: Rich component variety, consistent design system, clear page hierarchy

**Selected Subpages:**
1. Homepage (https://amazon.com) - Hero banners, product card grids, navigation
2. Category page (https://amazon.com/s?k=laptops) - Filters, product lists, pagination
3. Product detail page (https://amazon.com/dp/[ASIN]) - Image galleries, reviews, recommendations
4. Search results (https://amazon.com/s?k=wireless+mouse) - Different layout from category page

**Component types**: Navigation bar, search bar, product cards, filters, buttons, breadcrumbs, image galleries, review components, recommendation sections, footer, badges, pricing displays, etc. (20+ types)

### Example 2: Medium.com
**Domain Type**: News/Media  
**Why selected**: Card-based design, author profiles, varied content layouts

**Selected Subpages:**
1. Homepage (https://medium.com) - Article cards in feed, hero section
2. Publication page (https://medium.com/publication-name) - Publication header, article grid
3. Article detail (https://medium.com/@author/article-title) - Article layout, author card, related articles

**Component types**: Navigation, article cards, author profiles, hero sections, tag lists, reading progress, social sharing, comments, recommended articles, footer (15+ types)

### Example 3: GitHub.com
**Domain Type**: Technology/Software  
**Why selected**: Complex data displays, diverse interactive components, professional design

**Selected Subpages:**
1. Homepage (https://github.com) - Hero, feature sections, CTA buttons
2. Repository page (https://github.com/user/repo) - File tree, README, statistics
3. Explore page (https://github.com/explore) - Repository cards, trending lists

**Component types**: Navigation, tabs, file tree, code blocks, statistics cards, user avatars, labels, buttons, search, filters, repository cards (18+ types)

## 9. Examples of Poor Selections (Avoid These)

### ❌ Example 1: Craigslist.org
**Why NOT selected**: Extremely minimal design, outdated UI, very few component types, primarily text links

### ❌ Example 2: Reddit.com (old.reddit.com)
**Why NOT selected**: While new Reddit is acceptable, old Reddit has minimal visual components and outdated design patterns

### ❌ Example 3: Simple landing pages
**Why NOT selected**: Single-page sites with only a few sections, minimal component diversity, no multi-page structure

### ❌ Example 4: Heavily paywalled news sites
**Why NOT selected**: Cannot access enough content without subscription to properly evaluate and annotate

## 10. Common Mistakes to Avoid

❌ Selecting websites with insufficient complexity
❌ Choosing websites you personally like rather than objectively evaluating
❌ Selecting too many similar websites (e.g., 5 different e-commerce sites)
❌ Including subpages that are nearly identical in layout
❌ Forgetting to document URLs and page types
❌ Selecting websites that require login for basic content
❌ Choosing websites with outdated or poorly structured designs
❌ Not verifying technical accessibility before finalizing selection

## 11. Handling Edge Cases

### Case 1: Website has good components but requires login
**Solution**: If you can access 2-3 pages without login (homepage, about, landing pages), and they showcase sufficient component diversity, the website may still be selected. Document the limitation.

### Case 2: Website design changes frequently
**Solution**: Take screenshots immediately after selection. Document the date of selection. If major redesign occurs during annotation, flag for team discussion.

### Case 3: Disagreement on design complexity
**Solution**: Count distinct component types objectively. If count is 10+, it meets the threshold. Use the component categories from Guideline 1 as reference.

### Case 4: Cannot find 5 distinct subpages
**Solution**: Selecting 2-3 highly diverse subpages is acceptable if they collectively showcase rich component variety. Prioritize quality over quantity.

### Case 5: Website has both simple and complex pages
**Solution**: Focus selection on the complex pages that showcase the design system. Avoid the overly simple pages.

## 12. Collaborative Selection Process

If multiple annotators are selecting websites:

1. **Divide the Top 500 list**: Each annotator evaluates different segments
2. **Share selections**: Review each other's choices before finalizing
3. **Avoid duplicates**: Coordinate to ensure diversity across all selections
4. **Discuss borderline cases**: If uncertain about a website, bring to team discussion
5. **Maintain a shared registry**: Document all selected websites to prevent overlap

## 13. Timeline Recommendation

Per website:
- Initial evaluation: 3-5 minutes
- Deep evaluation: 10-15 minutes  
- Subpage selection: 5-10 minutes
- Documentation: 5 minutes
- **Total: 25-45 minutes per website**

Plan accordingly based on your target number of websites.

## 14. Final Deliverable

For each selected website, provide:

```markdown
## Website [Index]

**Name**: [Website Name]
**Homepage URL**: [URL]
**Domain Type**: [Category]
**Design Complexity**: [High/Medium/Moderate]
**Selection Date**: [Date]

**Subpages Selected**:
1. **Page Type**: [Homepage/Category/Detail/etc.]
   - URL: [Full URL]
   - Key Components: [Brief list]
   
2. **Page Type**: [Type]
   - URL: [Full URL]
   - Key Components: [Brief list]

[Repeat for all subpages]

**Rationale**: [1-2 sentences explaining why this website was selected]

**Estimated Component Types**: [Number]

**Notes**: [Any special considerations, limitations, or observations]

**Screenshots**: [Attached or referenced]
```

---

**Remember**: The goal is to select websites that will provide rich, diverse, and representative component examples for the annotation dataset. When in doubt, prioritize component diversity and design quality over personal preferences or website popularity.