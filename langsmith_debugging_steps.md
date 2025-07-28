# LangSmith Workspace Stats Mismatch - Specific Debugging Steps

## Overview
This document provides specific debugging steps to investigate the mismatch between the sidebar count (55) and table display (5) of tracing projects in LangSmith.

## 1. Comparing API Responses Between /current/stats and Projects Endpoints

### Step 1.1: Capture API Responses
```bash
# Capture the workspace stats endpoint response
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     -H "Content-Type: application/json" \
     "${LANGSMITH_API_URL}/workspaces/current/stats" \
     > workspace_stats_response.json

# Capture the projects/sessions endpoint response
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     -H "Content-Type: application/json" \
     "${LANGSMITH_API_URL}/sessions" \
     > projects_response.json

# Alternative projects endpoint (if different)
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     -H "Content-Type: application/json" \
     "${LANGSMITH_API_URL}/projects" \
     > projects_alt_response.json
```

### Step 1.2: Analyze Response Structure
```bash
# Examine the stats response structure
jq '.' workspace_stats_response.json

# Look specifically for tracer_session_count
jq '.tracer_session_count' workspace_stats_response.json

# Count actual projects in the response
jq '. | length' projects_response.json

# If projects are nested in a data field
jq '.data | length' projects_response.json
```

### Step 1.3: Compare Key Fields
Create a comparison script:
```python
import json

# Load responses
with open('workspace_stats_response.json') as f:
    stats = json.load(f)

with open('projects_response.json') as f:
    projects = json.load(f)

print(f"Stats endpoint tracer_session_count: {stats.get('tracer_session_count', 'Not found')}")
print(f"Projects endpoint count: {len(projects.get('data', projects))}")
print(f"Difference: {stats.get('tracer_session_count', 0) - len(projects.get('data', projects))}")
```

## 2. Checking for Pagination/Filtering Differences

### Step 2.1: Test Pagination Parameters
```bash
# Test projects endpoint with different pagination parameters
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     "${LANGSMITH_API_URL}/sessions?limit=100&offset=0" \
     > projects_page1.json

curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     "${LANGSMITH_API_URL}/sessions?limit=100&offset=100" \
     > projects_page2.json

# Check if there are more pages
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     "${LANGSMITH_API_URL}/sessions?limit=1000" \
     > projects_all.json
```

### Step 2.2: Analyze Pagination Response
```python
import json

# Check pagination metadata
with open('projects_page1.json') as f:
    page1 = json.load(f)

# Look for pagination fields
print("Pagination fields:")
for key in ['total', 'count', 'has_more', 'next_page', 'total_count']:
    if key in page1:
        print(f"  {key}: {page1[key]}")

# Count total across all pages
total_projects = 0
for page_file in ['projects_page1.json', 'projects_page2.json']:
    try:
        with open(page_file) as f:
            page_data = json.load(f)
            projects_in_page = len(page_data.get('data', page_data))
            total_projects += projects_in_page
            print(f"{page_file}: {projects_in_page} projects")
    except FileNotFoundError:
        break

print(f"Total projects across pages: {total_projects}")
```

### Step 2.3: Test Different Filter Parameters
```bash
# Test with different date ranges
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     "${LANGSMITH_API_URL}/sessions?start_time=2024-01-01&end_time=2024-12-31" \
     > projects_filtered_date.json

# Test with different status filters
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     "${LANGSMITH_API_URL}/sessions?status=active" \
     > projects_active.json

# Test without any filters
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     "${LANGSMITH_API_URL}/sessions" \
     > projects_unfiltered.json
```

## 3. Verifying Permission-Based Visibility

### Step 3.1: Test with Different User Contexts
```bash
# If you have access to different API keys/users, test with each
export LANGSMITH_API_KEY_USER1="your_user1_key"
export LANGSMITH_API_KEY_ADMIN="your_admin_key"

# Test stats endpoint with different users
curl -H "Authorization: Bearer $LANGSMITH_API_KEY_USER1" \
     "${LANGSMITH_API_URL}/workspaces/current/stats" \
     > stats_user1.json

curl -H "Authorization: Bearer $LANGSMITH_API_KEY_ADMIN" \
     "${LANGSMITH_API_URL}/workspaces/current/stats" \
     > stats_admin.json

# Test projects endpoint with different users
curl -H "Authorization: Bearer $LANGSMITH_API_KEY_USER1" \
     "${LANGSMITH_API_URL}/sessions" \
     > projects_user1.json

curl -H "Authorization: Bearer $LANGSMITH_API_KEY_ADMIN" \
     "${LANGSMITH_API_URL}/sessions" \
     > projects_admin.json
```

### Step 3.2: Compare Permission-Based Results
```python
import json

def compare_user_responses(stats_file1, stats_file2, projects_file1, projects_file2, user1_name, user2_name):
    with open(stats_file1) as f:
        stats1 = json.load(f)
    with open(stats_file2) as f:
        stats2 = json.load(f)
    with open(projects_file1) as f:
        projects1 = json.load(f)
    with open(projects_file2) as f:
        projects2 = json.load(f)
    
    print(f"{user1_name} stats count: {stats1.get('tracer_session_count', 'N/A')}")
    print(f"{user2_name} stats count: {stats2.get('tracer_session_count', 'N/A')}")
    print(f"{user1_name} projects count: {len(projects1.get('data', projects1))}")
    print(f"{user2_name} projects count: {len(projects2.get('data', projects2))}")

# Run comparison
compare_user_responses('stats_user1.json', 'stats_admin.json', 
                      'projects_user1.json', 'projects_admin.json',
                      'Regular User', 'Admin User')
```

### Step 3.3: Check Workspace Permissions
```bash
# Get current user info
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     "${LANGSMITH_API_URL}/users/current" \
     > current_user.json

# Get workspace members and roles
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
     "${LANGSMITH_API_URL}/workspaces/current/members" \
     > workspace_members.json
```

## 4. Investigating Data Consistency Between Stats Counting Logic and Table Display Logic

### Step 4.1: Analyze Project Metadata
```python
import json
from collections import Counter
from datetime import datetime

with open('projects_response.json') as f:
    projects_data = json.load(f)

projects = projects_data.get('data', projects_data)

# Analyze project statuses
statuses = [p.get('status', 'unknown') for p in projects]
status_counts = Counter(statuses)
print("Project statuses:")
for status, count in status_counts.items():
    print(f"  {status}: {count}")

# Analyze creation dates
creation_dates = []
for p in projects:
    if 'created_at' in p:
        try:
            date = datetime.fromisoformat(p['created_at'].replace('Z', '+00:00'))
            creation_dates.append(date)
        except:
            pass

if creation_dates:
    print(f"\nDate range:")
    print(f"  Earliest: {min(creation_dates)}")
    print(f"  Latest: {max(creation_dates)}")

# Check for archived/hidden projects
archived_count = sum(1 for p in projects if p.get('archived', False))
hidden_count = sum(1 for p in projects if p.get('hidden', False))
print(f"\nSpecial statuses:")
print(f"  Archived: {archived_count}")
print(f"  Hidden: {hidden_count}")
```

### Step 4.2: Cross-Reference with Stats Breakdown
```python
import json

# Load both responses
with open('workspace_stats_response.json') as f:
    stats = json.load(f)

with open('projects_response.json') as f:
    projects_data = json.load(f)

# Look for detailed breakdowns in stats
print("Stats response fields:")
for key, value in stats.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value}")

# Check if stats include different types of sessions/projects
if 'session_breakdown' in stats:
    print("\nSession breakdown:")
    for key, value in stats['session_breakdown'].items():
        print(f"  {key}: {value}")
```

### Step 4.3: Identify Discrepancy Patterns
```python
import json

def analyze_discrepancy():
    with open('workspace_stats_response.json') as f:
        stats = json.load(f)
    
    with open('projects_response.json') as f:
        projects_data = json.load(f)
    
    stats_count = stats.get('tracer_session_count', 0)
    projects_count = len(projects_data.get('data', projects_data))
    discrepancy = stats_count - projects_count
    
    print(f"Analysis Summary:")
    print(f"  Stats endpoint count: {stats_count}")
    print(f"  Projects endpoint count: {projects_count}")
    print(f"  Discrepancy: {discrepancy}")
    print(f"  Discrepancy percentage: {(discrepancy/stats_count)*100:.1f}%")
    
    # Hypotheses based on discrepancy size
    if discrepancy > 0:
        print(f"\nPossible explanations for {discrepancy} missing projects:")
        print("  - Projects are archived/hidden in the table view")
        print("  - Permission-based filtering in the table")
        print("  - Stats counting deleted projects that no longer appear")
        print("  - Different time ranges between endpoints")
        print("  - Caching issues in the stats endpoint")
    
    return discrepancy

analyze_discrepancy()
```

## Execution Checklist

- [ ] **API Response Comparison**
  - [ ] Captured both endpoint responses
  - [ ] Analyzed response structures
  - [ ] Compared key counting fields
  
- [ ] **Pagination/Filtering Analysis**
  - [ ] Tested pagination parameters
  - [ ] Verified total count across pages
  - [ ] Tested different filter combinations
  
- [ ] **Permission Verification**
  - [ ] Tested with different user roles (if available)
  - [ ] Compared permission-based results
  - [ ] Checked workspace member permissions
  
- [ ] **Data Consistency Investigation**
  - [ ] Analyzed project metadata and statuses
  - [ ] Cross-referenced with stats breakdown
  - [ ] Identified discrepancy patterns

## Expected Outcomes

After completing these debugging steps, you should have:

1. **Clear API Response Comparison**: Understanding of what each endpoint returns and how they differ
2. **Pagination Clarity**: Knowledge of whether pagination is causing the discrepancy
3. **Permission Impact**: Understanding of how user permissions affect the counts
4. **Data Consistency Insights**: Identification of the root cause of the 55 vs 5 mismatch

## Next Steps

Based on the debugging results:
- If pagination is the issue: Implement proper pagination handling in the frontend
- If permissions are the cause: Align permission logic between endpoints
- If data consistency is the problem: Report to LangSmith backend team with evidence
- If filtering differences exist: Standardize filtering logic across endpoints
