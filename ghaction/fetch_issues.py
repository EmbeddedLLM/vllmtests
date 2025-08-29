#!/usr/bin/env python3
"""
Script to fetch issues from vllm-project/vllm, create identical issues in your fork,
and validate GitHub Action labeling accuracy for ROCm-related content.
Updated to match the new GitHub Action configuration with searchIn properties.
"""

import os
import sys
import json
import argparse
import time
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Set, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class ROCmDetector:
    """ROCm keyword detection logic matching the updated GitHub Action."""
    
    def __init__(self):
        # This should match your updated GitHub Action configuration exactly
        self.config = {
            "rocm": {
                # Keyword search - matches whole words only (with word boundaries)
                "keywords": [
                    {
                        "term": "composable kernel",
                        "searchIn": "both"
                    },
                    {
                        "term": "rccl",
                        "searchIn": "body"  # only search in body
                    },
                    {
                        "term": "migraphx",
                        "searchIn": "title"  # only search in title
                    },
                    {
                        "term": "hipgraph",
                        "searchIn": "both"
                    },
                    {
                        "term": "ROCm System Management Interface",
                        "searchIn": "body"
                    },
                ],
                
                # Substring search - matches anywhere in text (partial matches)
                "substrings": [
                    {
                        "term": "VLLM_ROCM_",
                        "searchIn": "both"
                    },
                    {
                        "term": "aiter",
                        "searchIn": "title"
                    },
                    {
                        "term": "rocm",
                        "searchIn": "title"
                    },
                    {
                        "term": "amd",
                        "searchIn": "title"
                    },
                    {
                        "term": "hip-",
                        "searchIn": "both"
                    },
                    {
                        "term": "gfx",
                        "searchIn": "both"
                    },
                    {
                        "term": "cdna",
                        "searchIn": "both"
                    },
                    {
                        "term": "rdna",
                        "searchIn": "both"
                    },
                    {
                        "term": "torch_hip",
                        "searchIn": "body"  # only in body
                    },
                    {
                        "term": "_hip",
                        "searchIn": "both"
                    },
                    {
                        "term": "hip_",
                        "searchIn": "both"
                    },
                    # ROCm tools and libraries
                    {
                        "term": "hipify",
                        "searchIn": "both"
                    },
                ],
                
                # Regex patterns - for complex pattern matching
                "regexPatterns": [
                    {
                        "pattern": r"\bmi\d{3}[a-z]*\b",
                        "description": "AMD GPU names (mi + 3 digits + optional letters)",
                        "flags": "gi",
                        "searchIn": "both"  # "title", "body", or "both"
                    }
                ],
            }
        }
    
    def create_search_regex(self, term: str, search_type: str) -> re.Pattern:
        """Create regex based on search type."""
        escaped_term = re.escape(term)
        if search_type == 'keyword':
            return re.compile(rf'\b{escaped_term}\b', re.IGNORECASE)
        elif search_type == 'substring':
            return re.compile(escaped_term, re.IGNORECASE)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def find_matching_terms_with_lines(self, text: str, search_terms: List[Union[str, Dict]], 
                                     search_type: str = 'keyword', search_location: str = '') -> List[Dict]:
        """Find matching terms in text with line information."""
        matches = []
        lines = text.split('\n')
        
        for term_config in search_terms:
            # Handle different input formats (string or object)
            if isinstance(term_config, str):
                term = term_config
                search_in = 'both'  # default
                pattern = None
                description = None
                flags = None
            else:
                term = term_config.get('term')
                search_in = term_config.get('searchIn', 'both')
                pattern = term_config.get('pattern')
                description = term_config.get('description')
                flags = term_config.get('flags')
            
            # Skip if this term shouldn't be searched in the current location
            if search_in != 'both' and search_in != search_location:
                continue
            
            # Create appropriate regex
            if search_type == 'regex':
                flags_int = re.IGNORECASE if flags and 'i' in flags.lower() else 0
                regex = re.compile(pattern, flags_int)
            else:
                regex = self.create_search_regex(term, search_type)
            
            term_matches = []
            
            # Check each line for matches
            for line_idx, line in enumerate(lines, 1):
                line_matches = regex.findall(line)
                for match in line_matches:
                    # Show context around the match in the line
                    if len(line) > 100:
                        match_pos = line.lower().find(match.lower())
                        if match_pos >= 0:
                            context_start = max(0, match_pos - 30)
                            context_end = match_pos + len(match) + 30
                            context = line[context_start:context_end] + '...'
                        else:
                            context = line.strip()
                    else:
                        context = line.strip()
                    
                    term_matches.append({
                        "match": match,
                        "line_number": line_idx,
                        "line_content": line.strip(),
                        "search_type": search_type,
                        "search_location": search_location,
                        "original_term": term or pattern,
                        "description": description,
                        "context": context
                    })
            
            if term_matches:
                matches.append({
                    "term": term or (description or pattern),
                    "search_type": search_type,
                    "search_location": search_location,
                    "search_in": search_in,
                    "pattern": pattern,
                    "matches": term_matches,
                    "count": len(term_matches)
                })
        
        return matches
    
    def find_matching_terms(self, title: str, body: str) -> Dict:
        """Find ROCm-related terms in title and body with detailed match information."""
        config = self.config["rocm"]
        all_matches = []
        
        keywords = config.get("keywords", [])
        substrings = config.get("substrings", [])
        regex_patterns = config.get("regexPatterns", [])
        
        # Search in title
        if title.strip():
            title_keyword_matches = self.find_matching_terms_with_lines(title, keywords, 'keyword', 'title')
            title_substring_matches = self.find_matching_terms_with_lines(title, substrings, 'substring', 'title')
            title_regex_matches = self.find_matching_terms_with_lines(title, regex_patterns, 'regex', 'title')
            
            all_matches.extend(title_keyword_matches)
            all_matches.extend(title_substring_matches)
            all_matches.extend(title_regex_matches)
        
        # Search in body
        if body.strip():
            body_keyword_matches = self.find_matching_terms_with_lines(body, keywords, 'keyword', 'body')
            body_substring_matches = self.find_matching_terms_with_lines(body, substrings, 'substring', 'body')
            body_regex_matches = self.find_matching_terms_with_lines(body, regex_patterns, 'regex', 'body')
            
            all_matches.extend(body_keyword_matches)
            all_matches.extend(body_substring_matches)
            all_matches.extend(body_regex_matches)
        
        total_matches = sum(match_group["count"] for match_group in all_matches)
        
        return {
            "should_have_label": total_matches > 0,
            "matches": all_matches,
            "total_matches": total_matches
        }
    
    def analyze_issue(self, issue: Dict) -> Dict:
        """Analyze an issue for ROCm content."""
        title = issue.get('title', '') or ''
        body = issue.get('body', '') or ''
        
        analysis = self.find_matching_terms(title, body)
        analysis.update({
            "issue_number": issue.get('number'),
            "issue_title": title,
            "issue_url": issue.get('html_url', ''),
            "current_labels": [label['name'] for label in issue.get('labels', [])],
            "has_rocm_label": 'rocm' in [label['name'].lower() for label in issue.get('labels', [])]
        })
        
        return analysis

class GitHubIssueManager:
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        
        # Set up retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'vllm-issue-tester/1.0'
        })
    
    def test_connection(self):
        """Test the GitHub API connection and token."""
        response = self.session.get("https://api.github.com/user")
        if response.status_code == 200:
            user = response.json()
            print(f"✅ Connected to GitHub API as: {user.get('login', 'Unknown')}")
            return True
        else:
            print(f"❌ GitHub API connection failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    def fetch_issues(self, owner: str, repo: str, label: Optional[str] = None, 
                    since: Optional[datetime] = None, limit: int = 10, state: str = 'all') -> List[Dict]:
        """Fetch issues from a GitHub repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            'state': state,
            'per_page': min(limit, 100),
            'sort': 'created',
            'direction': 'desc'
        }
        
        if label:
            params['labels'] = label
        
        if since:
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
            params['since'] = since.isoformat()
        
        print(f"Fetching issues from {owner}/{repo}...")
        if label:
            print(f"  Filtering by label: {label}")
        if since:
            print(f"  Since: {since.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        all_issues = []
        page = 1
        
        while len(all_issues) < limit:
            params['page'] = page
            response = self.session.get(url, params=params)
            
            if response.status_code != 200:
                print(f"Error fetching issues: {response.status_code}")
                print(f"Response: {response.text}")
                break
            
            issues = response.json()
            if not issues:
                break
            
            # Filter out pull requests
            issues_only = [issue for issue in issues if 'pull_request' not in issue]
            all_issues.extend(issues_only)
            print(f"  Fetched page {page}: {len(issues_only)} issues")
            
            if len(issues) < params['per_page']:
                break
            
            page += 1
        
        all_issues = all_issues[:limit]
        print(f"Total issues fetched: {len(all_issues)}")
        return all_issues
    
    def get_issue(self, owner: str, repo: str, issue_number: int) -> Optional[Dict]:
        """Get a specific issue by number."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching issue #{issue_number}: {response.status_code}")
            return None
    
    def create_issue(self, owner: str, repo: str, original_issue: Dict) -> Optional[Dict]:
        """Create an issue in the target repository with identical title and body."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        
        title = original_issue['title']
        body = original_issue.get('body', '') or ''
        
        issue_data = {
            'title': title,
            'body': body,
        }
        
        print(f"Creating issue: {title[:60]}...")
        
        response = self.session.post(url, json=issue_data)
        
        if response.status_code == 201:
            created_issue = response.json()
            print(f"  ✅ Created: #{created_issue['number']} - {created_issue['html_url']}")
            return created_issue
        else:
            print(f"  ❌ Error creating issue: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    
    def wait_for_action(self, owner: str, repo: str, issue_number: int, 
                       timeout: int = 300, check_interval: int = 10) -> Optional[Dict]:
        """Wait for GitHub Action to process the issue and return updated issue."""
        print(f"  Waiting for GitHub Action to process issue #{issue_number}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            time.sleep(check_interval)
            issue = self.get_issue(owner, repo, issue_number)
            
            if issue:
                elapsed = int(time.time() - start_time)
                labels = [label['name'] for label in issue.get('labels', [])]
                print(f"    [{elapsed}s] Current labels: {labels}")
                
                # Check if action has run (look for any labels or wait a bit more)
                if labels or elapsed > 60:  # Either has labels or waited at least 1 minute
                    return issue
        
        print(f"  ⚠️  Timeout waiting for GitHub Action")
        return self.get_issue(owner, repo, issue_number)

class ValidationReporter:
    """Generate validation reports for ROCm labeling accuracy."""
    
    def __init__(self):
        self.detector = ROCmDetector()
    
    def validate_issue(self, issue: Dict, wait_for_action: bool = False, 
                      github_manager: Optional[GitHubIssueManager] = None,
                      repo_owner: str = None, repo_name: str = None) -> Dict:
        """Validate a single issue's ROCm labeling."""
        
        # If we should wait for action and have the necessary info
        if wait_for_action and github_manager and repo_owner and repo_name:
            issue = github_manager.wait_for_action(repo_owner, repo_name, issue['number'])
        
        analysis = self.detector.analyze_issue(issue)
        
        # Determine validation result
        expected_label = analysis["should_have_label"]
        has_label = analysis["has_rocm_label"]
        
        if expected_label and has_label:
            result = "CORRECT_POSITIVE"  # Should have label and does
        elif not expected_label and not has_label:
            result = "CORRECT_NEGATIVE"  # Should not have label and doesn't
        elif expected_label and not has_label:
            result = "FALSE_NEGATIVE"    # Should have label but doesn't
        else:  # not expected_label and has_label
            result = "FALSE_POSITIVE"    # Should not have label but does
        
        analysis["validation_result"] = result
        analysis["is_correct"] = result.startswith("CORRECT")
        
        return analysis
    
    def print_issue_analysis(self, analysis: Dict, verbose: bool = False):
        """Print detailed analysis of a single issue."""
        result = analysis["validation_result"]
        issue_num = analysis["issue_number"]
        title = analysis["issue_title"]
        
        # Result emoji and color
        result_info = {
            "CORRECT_POSITIVE": ("✅", "Correctly labeled"),
            "CORRECT_NEGATIVE": ("✅", "Correctly not labeled"),
            "FALSE_NEGATIVE": ("❌", "Missing ROCm label"),
            "FALSE_POSITIVE": ("⚠️", "Incorrectly labeled")
        }
        
        emoji, description = result_info[result]
        
        print(f"\n{emoji} Issue #{issue_num}: {title[:60]}...")
        print(f"   Result: {description}")
        print(f"   Current labels: {analysis['current_labels']}")
        print(f"   ROCm matches found: {analysis['total_matches']}")
        
        if verbose and analysis["matches"]:
            print("   Detected ROCm content:")
            for match_group in analysis["matches"]:
                search_in_text = match_group.get('search_in', 'both')
                location_text = match_group['search_location']
                print(f"     • {match_group['term']} ({match_group['search_type']}) in {location_text}: {match_group['count']} matches")
                print(f"       (configured to search in: {search_in_text})")
                
                if len(match_group['matches']) <= 3:  # Show details for few matches
                    for match in match_group['matches']:
                        print(f"       - Line {match['line_number']}: '{match['match']}'")
                        if match.get('description'):
                            print(f"         Description: {match['description']}")
        
        if analysis["issue_url"]:
            print(f"   URL: {analysis['issue_url']}")
    
    def generate_report(self, validations: List[Dict]) -> Dict:
        """Generate a summary report of validation results."""
        if not validations:
            return {"error": "No validations to report"}
        
        total = len(validations)
        correct = sum(1 for v in validations if v["is_correct"])
        
        results_count = {}
        for validation in validations:
            result = validation["validation_result"]
            results_count[result] = results_count.get(result, 0) + 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        return {
            "total_issues": total,
            "correct_predictions": correct,
            "accuracy_percentage": accuracy,
            "results_breakdown": results_count,
            "false_negatives": results_count.get("FALSE_NEGATIVE", 0),
            "false_positives": results_count.get("FALSE_POSITIVE", 0)
        }
    
    def print_report(self, report: Dict):
        """Print a formatted validation report."""
        if "error" in report:
            print(f"Report Error: {report['error']}")
            return
        
        print("\n" + "="*60)
        print("🔍 ROCm LABELING VALIDATION REPORT")
        print("="*60)
        
        print(f"Total Issues Analyzed: {report['total_issues']}")
        print(f"Correct Predictions: {report['correct_predictions']}")
        print(f"Accuracy: {report['accuracy_percentage']:.1f}%")
        
        print(f"\nDetailed Results:")
        breakdown = report['results_breakdown']
        print(f"  ✅ Correct Positive (should have + has label): {breakdown.get('CORRECT_POSITIVE', 0)}")
        print(f"  ✅ Correct Negative (should not have + no label): {breakdown.get('CORRECT_NEGATIVE', 0)}")
        print(f"  ❌ False Negative (should have + missing label): {breakdown.get('FALSE_NEGATIVE', 0)}")
        print(f"  ⚠️  False Positive (should not have + has label): {breakdown.get('FALSE_POSITIVE', 0)}")
        
        if report['false_negatives'] > 0:
            print(f"\n⚠️  {report['false_negatives']} issues are missing ROCm labels!")
        
        if report['false_positives'] > 0:
            print(f"⚠️  {report['false_positives']} issues are incorrectly labeled with ROCm!")

def parse_date(date_string: str) -> datetime:
    """Parse date string in various formats."""
    formats = ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_string, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date: {date_string}")

def main():
    parser = argparse.ArgumentParser(
        description="Fetch, create, and validate ROCm issue labeling with updated searchIn logic"
    )
    
    parser.add_argument('--token', help='GitHub personal access token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--source-repo', default='vllm-project/vllm', help='Source repository (default: vllm-project/vllm)')
    parser.add_argument('--target-repo', default='vllmellm/vllm', help='Target repository for issues (default: vllmellm/vllm)')
    parser.add_argument('--label', help='Filter issues by label (e.g., "rocm")')
    parser.add_argument('--since', help='Fetch issues created since this date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, help='Fetch issues from the last N days')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of issues to fetch (default: 5)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without creating issues')
    
    # Validation options
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing issues in target repo')
    parser.add_argument('--validate-source', action='store_true', help='Validate issues in source repo')
    parser.add_argument('--wait-for-action', action='store_true', help='Wait for GitHub Action to process created issues')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout for waiting for GitHub Action (seconds)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed match information')
    
    # Issue management
    parser.add_argument('--list-target', action='store_true', help='List recent issues in target repo')
    parser.add_argument('--close', help='Close issues in target repo (comma-separated issue numbers)')
    
    args = parser.parse_args()
    
    # Get GitHub token
    token = args.token or os.getenv('GITHUB_TOKEN')
    if not token:
        print("Error: GitHub token required. Use --token or set GITHUB_TOKEN environment variable.")
        sys.exit(1)
    
    # Parse repository names
    try:
        source_owner, source_repo = args.source_repo.split('/')
        target_owner, target_repo = args.target_repo.split('/')
    except ValueError:
        print("Error: Repository names must be in format 'owner/repo'")
        sys.exit(1)
    
    # Initialize components
    github = GitHubIssueManager(token)
    reporter = ValidationReporter()
    
    # Test connection
    if not github.test_connection():
        sys.exit(1)
    
    # Handle different modes
    if args.list_target:
        print(f"Recent issues in {args.target_repo}:")
        issues = github.fetch_issues(target_owner, target_repo, limit=20, state='open')
        for issue in issues:
            labels = [label['name'] for label in issue.get('labels', [])]
            label_str = f" [{', '.join(labels)}]" if labels else ""
            created = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
            print(f"  #{issue['number']}: {issue['title'][:60]}...{label_str}")
            print(f"    Created: {created.strftime('%Y-%m-%d %H:%M')} | State: {issue['state']}")
        return
    
    if args.close:
        try:
            issue_numbers = [int(num.strip()) for num in args.close.split(',')]
            for issue_number in issue_numbers:
                issue_url = f"https://api.github.com/repos/{target_owner}/{target_repo}/issues/{issue_number}"
                response = github.session.patch(issue_url, json={'state': 'closed'})
                if response.status_code == 200:
                    print(f"  ✅ Closed: #{issue_number}")
                else:
                    print(f"  ❌ Error closing #{issue_number}: {response.status_code}")
        except ValueError:
            print("Error: --close requires comma-separated issue numbers")
            sys.exit(1)
        return
    
    if args.validate_only:
        print(f"Validating existing issues in {args.target_repo}...")
        issues = github.fetch_issues(target_owner, target_repo, limit=args.limit, state='open')
        
        validations = []
        for issue in issues:
            validation = reporter.validate_issue(issue)
            validations.append(validation)
            reporter.print_issue_analysis(validation, args.verbose)
        
        report = reporter.generate_report(validations)
        reporter.print_report(report)
        return
    
    if args.validate_source:
        print(f"Validating issues in source repo {args.source_repo}...")
        
        # Parse date filters
        since_date = None
        if args.since:
            since_date = parse_date(args.since)
        elif args.days:
            since_date = datetime.now(timezone.utc) - timedelta(days=args.days)
        
        issues = github.fetch_issues(source_owner, source_repo, label=args.label, 
                                   since=since_date, limit=args.limit)
        
        validations = []
        for issue in issues:
            validation = reporter.validate_issue(issue)
            validations.append(validation)
            reporter.print_issue_analysis(validation, args.verbose)
        
        report = reporter.generate_report(validations)
        reporter.print_report(report)
        return
    
    # Main workflow: fetch, create, and optionally validate
    # Parse date filters
    since_date = None
    if args.since:
        since_date = parse_date(args.since)
    elif args.days:
        since_date = datetime.now(timezone.utc) - timedelta(days=args.days)
    
    # Fetch issues from source
    print("Fetching issues from source repository...")
    issues = github.fetch_issues(source_owner, source_repo, label=args.label, 
                               since=since_date, limit=args.limit)
    
    if not issues:
        print("No issues found matching the criteria.")
        return
    
    print(f"\nFound {len(issues)} issues to process:")
    for i, issue in enumerate(issues, 1):
        labels = [label['name'] for label in issue.get('labels', [])]
        label_str = f" [{', '.join(labels)}]" if labels else ""
        created = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
        print(f"  {i}. #{issue['number']}: {issue['title'][:60]}...{label_str}")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would create {len(issues)} issues in {args.target_repo}")
        
        # Show what our detector thinks about these issues
        print("\nROCm detection analysis:")
        for issue in issues:
            analysis = reporter.detector.analyze_issue(issue)
            should_label = "YES" if analysis["should_have_label"] else "NO"
            matches = analysis["total_matches"]
            print(f"  #{issue['number']}: Should have ROCm label: {should_label} ({matches} matches)")
            
            if args.verbose and analysis["matches"]:
                for match_group in analysis["matches"]:
                    search_in = match_group.get('search_in', 'both')
                    location = match_group['search_location']
                    print(f"    • {match_group['term']} ({match_group['search_type']}) in {location}: {match_group['count']} matches (searchIn: {search_in})")
        return
    
    # Create issues and optionally validate
    print(f"\nCreating issues in {args.target_repo}...")
    created_issues = []
    
    for issue in issues:
        created_issue = github.create_issue(target_owner, target_repo, issue)
        if created_issue:
            created_issues.append(created_issue)
    
    print(f"\n✅ Successfully created {len(created_issues)} issues!")
    
    # Validate if requested
    if args.wait_for_action and created_issues:
        print(f"\n🔍 Validating GitHub Action labeling...")
        validations = []
        
        for created_issue in created_issues:
            validation = reporter.validate_issue(
                created_issue, 
                wait_for_action=True,
                github_manager=github,
                repo_owner=target_owner,
                repo_name=target_repo
            )
            validations.append(validation)
            reporter.print_issue_analysis(validation, args.verbose)
        
        report = reporter.generate_report(validations)
        reporter.print_report(report)
    
    print(f"\nVisit your repository: https://github.com/{args.target_repo}/issues")

if __name__ == "__main__":
    main()