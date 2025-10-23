# Requirements Document

## Introduction

The Baseline Sidekick is a VS Code extension that helps developers identify web platform features that are not part of the Baseline standard and provides actionable suggestions for compatibility improvements. The extension analyzes CSS, JavaScript, and HTML files to detect non-Baseline features, provides educational hover information, and offers automated refactoring suggestions to improve cross-browser compatibility.

## Requirements

### Requirement 1

**User Story:** As a web developer, I want to see visual indicators when I use non-Baseline web features, so that I can make informed decisions about browser compatibility.

#### Acceptance Criteria

1. WHEN a file contains CSS properties not supported by Baseline THEN the system SHALL display diagnostic warnings with underlines in the editor
2. WHEN a file contains JavaScript APIs not supported by Baseline THEN the system SHALL display diagnostic warnings with underlines in the editor  
3. WHEN a file contains HTML elements or attributes not supported by Baseline THEN the system SHALL display diagnostic warnings with underlines in the editor
4. WHEN the extension activates THEN the system SHALL load and cache the web-features dataset for fast lookups
5. WHEN a file is opened or edited THEN the system SHALL automatically analyze it and update diagnostics within 500ms

### Requirement 2

**User Story:** As a web developer, I want detailed information about non-Baseline features when I hover over them, so that I can understand compatibility implications and find alternatives.

#### Acceptance Criteria

1. WHEN I hover over a non-Baseline feature THEN the system SHALL display a rich tooltip with feature information
2. WHEN displaying hover information THEN the system SHALL include the feature name and "Not Baseline Supported" badge
3. WHEN displaying hover information THEN the system SHALL show the specific reason for non-support (e.g., "Not supported by Safari 16.3")
4. WHEN displaying hover information THEN the system SHALL provide clickable links to MDN documentation and CanIUse.com
5. IF a suggested alternative exists THEN the system SHALL display it in the hover tooltip

### Requirement 3

**User Story:** As a web developer, I want automated refactoring suggestions for non-Baseline features, so that I can quickly fix compatibility issues without manual research.

#### Acceptance Criteria

1. WHEN I trigger code actions on a non-Baseline feature THEN the system SHALL provide relevant refactoring options
2. WHEN refactoring CSS float properties THEN the system SHALL offer to convert to Flexbox layout
3. WHEN refactoring XMLHttpRequest usage THEN the system SHALL offer to convert to modern fetch API
4. WHEN refactoring Array.at() usage THEN the system SHALL offer bracket notation with length check alternative
5. WHEN multiple refactoring options exist THEN the system SHALL mark the best option as preferred
6. WHEN applying a refactoring THEN the system SHALL make the code changes automatically via workspace edit

### Requirement 4

**User Story:** As a web developer, I want to audit my entire project for Baseline compatibility, so that I can get a comprehensive overview of all compatibility issues.

#### Acceptance Criteria

1. WHEN I run the workspace audit command THEN the system SHALL scan all JavaScript, CSS, and HTML files in the workspace
2. WHEN running the audit THEN the system SHALL exclude node_modules and other ignored directories
3. WHEN scanning files THEN the system SHALL display a progress indicator to the user
4. WHEN the audit completes THEN the system SHALL generate a Markdown report grouping issues by file
5. WHEN the report is ready THEN the system SHALL open it in a new editor tab for review
6. WHEN scanning large projects THEN the system SHALL complete the audit within reasonable time limits

### Requirement 5

**User Story:** As a web developer, I want the extension to accurately parse different file types, so that it can detect web platform features across my entire codebase.

#### Acceptance Criteria

1. WHEN parsing CSS files THEN the system SHALL use PostCSS to extract property names from declarations
2. WHEN parsing JavaScript files THEN the system SHALL use Babel parser to extract API usage from AST nodes
3. WHEN parsing HTML files THEN the system SHALL use parse5 to extract element and attribute names
4. WHEN analyzing JavaScript THEN the system SHALL detect MemberExpression patterns (e.g., navigator.clipboard)
5. WHEN analyzing JavaScript THEN the system SHALL detect CallExpression patterns on global objects
6. IF parsing fails for a file THEN the system SHALL handle errors gracefully without crashing the extension

### Requirement 6

**User Story:** As a web developer, I want fast and reliable access to Baseline compatibility data, so that the extension doesn't slow down my development workflow.

#### Acceptance Criteria

1. WHEN the extension activates THEN the system SHALL load the web-features dataset only once
2. WHEN looking up feature compatibility THEN the system SHALL return results within 10ms
3. WHEN checking if a feature is Baseline supported THEN the system SHALL handle cases where the feature doesn't exist
4. WHEN constructing MDN URLs THEN the system SHALL generate correct links from feature data
5. IF the web-features data is unavailable THEN the system SHALL handle the error gracefully and inform the user