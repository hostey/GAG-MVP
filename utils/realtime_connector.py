# utils/realtime_connector.py
"""
Real-time data ingestion for the GAGS Disinformation Simulator.

Supported sources
─────────────────
  • Twitter/X  — via Tweepy (Bearer Token, v2 API)
  • Reddit      — via PRAW  (client_id + client_secret)
  • NewsAPI     — via newsapi-python (API key)
  • RSS feeds   — via feedparser (no key required)
  • Demo mode   — fully synthetic data, no keys needed

Each source returns a standardised list of NarrativePost objects
that the simulation pipeline consumes directly.

Install dependencies
────────────────────
  pip install tweepy praw newsapi-python feedparser requests
"""

from __future__ import annotations

import os
import time
import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Universal data model
# ─────────────────────────────────────────────────────────────

@dataclass
class NarrativePost:
    """
    One piece of content that will be assigned to a network node.

    Fields
    ------
    post_id       : stable unique identifier
    text          : raw text content
    source        : 'twitter' | 'reddit' | 'newsapi' | 'rss' | 'demo'
    author        : screen-name or username
    created_at    : UTC timestamp
    engagement    : combined likes / upvotes / shares
    reach_proxy   : follower / subscriber count (0 if unknown)
    url           : original URL
    topic_tags    : list of hashtags, subreddit, topic keywords
    credibility   : 0.0 – 1.0 (higher = more credible source signal)
    virality_score: 0.0 – 1.0 (derived from engagement + reach)
    raw            : original API response dict
    """
    post_id:       str
    text:          str
    source:        str
    author:        str           = "unknown"
    created_at:    datetime      = field(default_factory=lambda: datetime.now(timezone.utc))
    engagement:    int           = 0
    reach_proxy:   int           = 0
    url:           str           = ""
    topic_tags:    List[str]     = field(default_factory=list)
    credibility:   float         = 0.5
    virality_score: float        = 0.0
    raw:           Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "post_id":       self.post_id,
            "text":          self.text,
            "source":        self.source,
            "author":        self.author,
            "created_at":    self.created_at.isoformat(),
            "engagement":    self.engagement,
            "reach_proxy":   self.reach_proxy,
            "url":           self.url,
            "topic_tags":    ", ".join(self.topic_tags),
            "credibility":   round(self.credibility, 3),
            "virality_score": round(self.virality_score, 3),
        }


def _virality(engagement: int, reach_proxy: int) -> float:
    """Normalise virality to [0, 1] using a log scale."""
    eng_log   = np.log1p(engagement)
    reach_log = np.log1p(reach_proxy)
    raw       = 0.6 * eng_log + 0.4 * reach_log
    return float(np.clip(raw / 20.0, 0.0, 1.0))   # 20 = saturation point


def _stable_id(source: str, native_id: str) -> str:
    return hashlib.md5(f"{source}:{native_id}".encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────
# Twitter / X  (Tweepy v4, API v2)
# ─────────────────────────────────────────────────────────────

class TwitterConnector:
    """
    Fetch recent tweets matching a query using the Twitter v2 search API.

    Required environment variables (or pass directly):
        TWITTER_BEARER_TOKEN
    """

    SOURCE = "twitter"

    def __init__(self, bearer_token: Optional[str] = None):
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import tweepy  # noqa
                self._client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    wait_on_rate_limit=True,
                )
            except ImportError:
                raise ImportError("Install tweepy: pip install tweepy")
        return self._client

    def fetch(
        self,
        query: str,
        max_results: int = 100,
        date_from: Optional[datetime] = None,
        date_to:   Optional[datetime] = None,
    ) -> List[NarrativePost]:
        """
        Search recent tweets.

        Parameters
        ----------
        query       : Twitter search query string, e.g. "misinformation -is:retweet"
        max_results : 10–100 per request (Twitter v2 limit per call)
        date_from   : start of date range (UTC). Defaults to 24 h ago.
        date_to     : end of date range (UTC). Defaults to now.
                      Twitter free tier only allows the past 7 days.
        """
        client = self._get_client()
        now    = datetime.now(timezone.utc)
        start  = date_from or (now - timedelta(hours=24))
        end    = date_to   or now

        # Ensure filter to reduce noise
        safe_query = f"({query}) lang:en -is:retweet"

        try:
            response = client.search_recent_tweets(
                query=safe_query,
                max_results=min(max_results, 100),
                start_time=start,
                end_time=end,
                tweet_fields=["created_at", "public_metrics",
                               "author_id", "entities"],
                expansions=["author_id"],
                user_fields=["public_metrics", "username"],
            )
        except Exception as exc:
            logger.warning(f"Twitter fetch failed: {exc}")
            return []

        if not response.data:
            return []

        # Build author lookup for follower counts
        users = {u.id: u for u in (response.includes.get("users") or [])}

        posts = []
        for tweet in response.data:
            pm       = tweet.public_metrics or {}
            author   = users.get(tweet.author_id)
            followers = author.public_metrics.get("followers_count", 0) if author else 0
            username  = author.username if author else str(tweet.author_id)
            engagement = (pm.get("like_count", 0)
                          + pm.get("retweet_count", 0)
                          + pm.get("reply_count", 0))
            hashtags = [t["tag"] for t in
                        (tweet.entities or {}).get("hashtags", [])]

            posts.append(NarrativePost(
                post_id       = _stable_id(self.SOURCE, str(tweet.id)),
                text          = tweet.text,
                source        = self.SOURCE,
                author        = username,
                created_at    = tweet.created_at or datetime.now(timezone.utc),
                engagement    = engagement,
                reach_proxy   = followers,
                url           = f"https://twitter.com/i/web/status/{tweet.id}",
                topic_tags    = hashtags,
                credibility   = min(1.0, np.log1p(followers) / 15),
                virality_score= _virality(engagement, followers),
                raw           = pm,
            ))

        return posts


# ─────────────────────────────────────────────────────────────
# Reddit  (PRAW)
# ─────────────────────────────────────────────────────────────

class RedditConnector:
    """
    Fetch hot / new posts from one or more subreddits.

    Required environment variables (or pass directly):
        REDDIT_CLIENT_ID
        REDDIT_CLIENT_SECRET
        REDDIT_USER_AGENT   (default: "GAGS-Disinfo-Simulator/1.0")
    """

    SOURCE = "reddit"

    def __init__(
        self,
        client_id: Optional[str]     = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str]    = None,
    ):
        self.client_id     = client_id     or os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET", "")
        self.user_agent    = user_agent    or os.getenv(
            "REDDIT_USER_AGENT", "GAGS-Disinfo-Simulator/1.0")
        self._reddit = None

    def _get_reddit(self):
        if self._reddit is None:
            try:
                import praw  # noqa
                self._reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
            except ImportError:
                raise ImportError("Install praw: pip install praw")
        return self._reddit

    def fetch(
        self,
        subreddits: List[str],
        mode: str = "hot",
        limit: int = 50,
        time_filter: str = "day",
        date_from: Optional[datetime] = None,
        date_to:   Optional[datetime] = None,
    ) -> List[NarrativePost]:
        """
        Parameters
        ----------
        subreddits  : list of subreddit names without r/ prefix
        mode        : listing mode — hot, new, or top
        limit       : posts per subreddit (fetch extra to allow date filtering)
        time_filter : applies only to 'top' mode: hour/day/week/month/year/all
        date_from   : keep posts on or after this UTC datetime
        date_to     : keep posts on or before this UTC datetime
        """
        reddit = self._get_reddit()
        now    = datetime.now(timezone.utc)
        cutoff_from = date_from or (now - timedelta(hours=48))
        cutoff_to   = date_to   or now
        posts  = []

        for sub_name in subreddits:
            try:
                sub = reddit.subreddit(sub_name)
                # Fetch more than requested so the date filter still
                # returns a useful number of posts
                fetch_limit = limit * 3
                if mode == "hot":
                    listing = sub.hot(limit=fetch_limit)
                elif mode == "new":
                    listing = sub.new(limit=fetch_limit)
                else:
                    listing = sub.top(limit=fetch_limit, time_filter=time_filter)

                for post in listing:
                    if post.stickied:
                        continue
                    post_dt = datetime.fromtimestamp(
                        post.created_utc, tz=timezone.utc)
                    # Apply date range filter
                    if post_dt < cutoff_from or post_dt > cutoff_to:
                        continue
                    engagement = post.score + post.num_comments
                    posts.append(NarrativePost(
                        post_id       = _stable_id(self.SOURCE, post.id),
                        text          = f"{post.title}. {post.selftext}"[:2000],
                        source        = self.SOURCE,
                        author        = str(post.author) if post.author else "deleted",
                        created_at    = post_dt,
                        engagement    = engagement,
                        reach_proxy   = sub.subscribers,
                        url           = f"https://reddit.com{post.permalink}",
                        topic_tags    = [sub_name] + post.title.split()[:5],
                        credibility   = min(1.0,
                                            np.log1p(sub.subscribers) / 20),
                        virality_score= _virality(engagement, sub.subscribers),
                        raw           = {"score": post.score,
                                         "num_comments": post.num_comments},
                    ))
            except Exception as exc:
                logger.warning(f"Reddit fetch failed for r/{sub_name}: {exc}")

        return posts


# ─────────────────────────────────────────────────────────────
# NewsAPI
# ─────────────────────────────────────────────────────────────

class NewsAPIConnector:
    """
    Fetch recent news articles matching a keyword query.

    Required environment variables (or pass directly):
        NEWSAPI_KEY
    """

    SOURCE = "newsapi"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from newsapi import NewsApiClient  # noqa
                self._client = NewsApiClient(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install newsapi-python: pip install newsapi-python")
        return self._client

    # Credibility lookup by domain (extensible)
    DOMAIN_CREDIBILITY: Dict[str, float] = {
        # ── Global ──────────────────────────────────────────────────────────
        "reuters.com": 0.95, "apnews.com": 0.93, "bbc.co.uk": 0.92,
        "bbc.com": 0.92, "nytimes.com": 0.85, "theguardian.com": 0.84,
        "washingtonpost.com": 0.83, "npr.org": 0.88, "politico.com": 0.75,
        "foxnews.com": 0.60, "breitbart.com": 0.30, "infowars.com": 0.10,
        "rt.com": 0.25, "sputniknews.com": 0.20,
        # ── Nigeria — mainstream ─────────────────────────────────────────────
        "premiumtimesng.com": 0.87,
        "punchng.com": 0.80,
        "vanguardngr.com": 0.78,
        "thecable.ng": 0.85,
        "dailypost.ng": 0.75,
        "thenationonlineng.net": 0.74,
        "pmnewsnigeria.com": 0.72,
        "saharareporters.com": 0.70,
        "channelstv.com": 0.82,
        "arise.tv": 0.78,
        # ── Nigeria — fact-checkers ──────────────────────────────────────────
        "factcheckhub.com": 0.93,
        "dubawa.org": 0.92,
        "factcheckafrica.net": 0.90,
        # ── Pan-African fact-checkers ────────────────────────────────────────
        "africacheck.org": 0.94,
        "pesacheck.org": 0.91,
        "allafrica.com": 0.72,
        # ── East Africa ─────────────────────────────────────────────────────
        "theeastafrican.co.ke": 0.82,
        "nation.africa": 0.81,
        "standardmedia.co.ke": 0.76,
        # ── West Africa ─────────────────────────────────────────────────────
        "ghanaweb.com": 0.70,
        "myjoyonline.com": 0.78,
        "graphic.com.gh": 0.80,
        # ── Southern Africa ──────────────────────────────────────────────────
        "dailymaverick.co.za": 0.90,
        "news24.com": 0.78,
        "groundup.org.za": 0.88,
        # ── Low-credibility / misinformation-prone ───────────────────────────
        "informationng.com": 0.35,
        "naijaloaded.com.ng": 0.30,
        "torizone.com": 0.28,
    }

    def _domain_credibility(self, url: str) -> float:
        for domain, score in self.DOMAIN_CREDIBILITY.items():
            if domain in url:
                return score
        return 0.5   # unknown domain → neutral

    def fetch(
        self,
        query: str,
        language: str = "en",
        max_articles: int = 100,
        date_from: Optional[datetime] = None,
        date_to:   Optional[datetime] = None,
    ) -> List[NarrativePost]:
        """
        Parameters
        ----------
        query        : search query string
        language     : ISO 639-1 language code (default 'en')
        max_articles : max results (NewsAPI free tier cap: 100/request)
        date_from    : start of date range (UTC). Defaults to 48 h ago.
                       NewsAPI free tier allows up to 1 month back.
        date_to      : end of date range (UTC). Defaults to now.
        """
        client = self._get_client()
        now    = datetime.now(timezone.utc)
        since  = (date_from or (now - timedelta(hours=48))).strftime(
            "%Y-%m-%dT%H:%M:%S")
        until  = (date_to or now).strftime("%Y-%m-%dT%H:%M:%S")

        try:
            resp = client.get_everything(
                q=query, language=language,
                from_param=since,
                to=until,
                sort_by="relevancy",
                page_size=min(max_articles, 100),
            )
        except Exception as exc:
            logger.warning(f"NewsAPI fetch failed: {exc}")
            return []

        posts = []
        for art in (resp.get("articles") or []):
            url  = art.get("url", "")
            text = " ".join(filter(None, [art.get("title"),
                                          art.get("description"),
                                          art.get("content")]))
            if not text.strip():
                continue

            pub_at = art.get("publishedAt", "")
            try:
                created = datetime.fromisoformat(
                    pub_at.replace("Z", "+00:00"))
            except Exception:
                created = datetime.now(timezone.utc)

            cred = self._domain_credibility(url)
            posts.append(NarrativePost(
                post_id       = _stable_id(self.SOURCE,
                                           url or text[:40]),
                text          = text[:2000],
                source        = self.SOURCE,
                author        = (art.get("source") or {}).get("name", "Unknown"),
                created_at    = created,
                engagement    = 0,    # NewsAPI doesn't expose engagement
                reach_proxy   = 0,
                url           = url,
                topic_tags    = [w.lower() for w in query.split()],
                credibility   = cred,
                virality_score= 1.0 - cred,   # low-cred articles spread faster
                raw           = {"source": art.get("source")},
            ))

        return posts


# ─────────────────────────────────────────────────────────────
# RSS / Atom feeds  (no API key required)
# ─────────────────────────────────────────────────────────────

# Curated list of RSS feeds relevant to misinformation research
DEFAULT_RSS_FEEDS: Dict[str, str] = {
    # ── Global fact-check & news ───────────────────────────────────────────────
    "Reuters Top News":        "https://feeds.reuters.com/reuters/topNews",
    "BBC News":                "http://feeds.bbci.co.uk/news/rss.xml",
    "AP News":                 "https://rsshub.app/apnews/topics/apf-topnews",
    "First Draft (disinfo)":   "https://firstdraftnews.org/feed/",
    "Poynter Fact-Check":      "https://www.poynter.org/ifcn/feed/",
    "PolitiFact":              "https://www.politifact.com/rss/factchecks/",
    "Snopes":                  "https://www.snopes.com/feed/",
    "EUvsDisinfo":             "https://euvsdisinfo.eu/feed/",
    "Full Fact":               "https://fullfact.org/feed/",
    "MediaBias/FactCheck":     "https://mediabiasfactcheck.com/feed/",

    # ── Nigeria — mainstream news ──────────────────────────────────────────────
    # Premium Times: Nigeria's leading investigative & breaking news outlet
    "Premium Times (Nigeria)":     "https://www.premiumtimesng.com/feed",
    # Vanguard: one of Nigeria's highest-circulation daily newspapers
    "Vanguard News":               "https://www.vanguardngr.com/feed",
    # The Punch: largest-circulation newspaper in Nigeria
    "The Punch (Nigeria)":         "https://punchng.com/feed/",
    # Daily Post: fast-breaking Nigeria & Africa news
    "Daily Post Nigeria":          "https://dailypost.ng/feed",
    # TheCable: independent investigative journalism from Nigeria
    "TheCable Nigeria":            "https://www.thecable.ng/feed",
    # Sahara Reporters: citizen journalism & anti-corruption reporting
    "Sahara Reporters":            "https://saharareporters.com/articles/rss-feed",
    # The Nation: major Lagos-based daily newspaper
    "The Nation (Nigeria)":        "https://thenationonlineng.net/feed/",
    # PM News Nigeria: Lagos-based breaking news
    "PM News Nigeria":             "https://pmnewsnigeria.com/feed",

    # ── Nigeria — fact-checking organisations ─────────────────────────────────
    # FactCheckHub: verification arm of the International Centre for
    # Investigative Reporting (ICIR), IFCN-certified
    "FactCheckHub (Nigeria)":      "https://factcheckhub.com/feed/",
    # DUBAWA: West African fact-checking project by CJID,
    # active in Nigeria, Ghana, Sierra Leone, Liberia & The Gambia
    "DUBAWA (West Africa)":        "https://dubawa.org/feed/",
    # TheCable Fact-Check: dedicated fact-check vertical of TheCable
    "TheCable Fact-Check":         "https://factcheck.thecable.ng/feed/",
    # FactCheckAfrica: Nigeria-based, runs MyAIFactChecker tool
    "FactCheck Africa":            "https://factcheckafrica.net/feed/",

    # ── Pan-African fact-checking & journalism ────────────────────────────────
    # Africa Check: continent's leading independent fact-checker,
    # founded 2012, offices in Nigeria, South Africa, Kenya, Senegal
    "Africa Check":                "https://africacheck.org/feed/",
    # PesaCheck: Code for Africa's financial fact-checking across 12 countries
    "PesaCheck":                   "https://pesacheck.org/feed/",
    # AFP Fact Check Africa: Agence France-Presse African fact-check desk
    "AFP Fact Check Africa":       "https://factcheck.afp.com/list/africa/rss",
    # AllAfrica: aggregator of 130+ African news sources
    "AllAfrica":                   "https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf",
    # The Continent: pan-African weekly news digest
    "The Continent":               "https://thecontinent.org/feed/",

    # ── East Africa ───────────────────────────────────────────────────────────
    # The East African: regional newspaper covering Kenya, Uganda, Tanzania
    "The East African":            "https://www.theeastafrican.co.ke/tea/rss",
    # Nation Africa (Kenya): Kenya's largest media house
    "Nation Africa (Kenya)":       "https://nation.africa/kenya/rss.xml",

    # ── West Africa ───────────────────────────────────────────────────────────
    # Ghana Web: Ghana's most visited news site
    "Ghana Web":                   "https://www.ghanaweb.com/GhanaHomePage/rss/news.xml",
    # Joy News (Ghana): multimedia news from Multimedia Group
    "Joy News (Ghana)":            "https://www.myjoyonline.com/feed/",

    # ── Southern Africa ───────────────────────────────────────────────────────
    # Daily Maverick (South Africa): award-winning investigative journalism
    "Daily Maverick (SA)":         "https://www.dailymaverick.co.za/feed/",
    # News24 (South Africa): South Africa's most-read news site
    "News24 (SA)":                 "https://feeds.news24.com/articles/news24/TopStories/rss",

    # ── African development & governance ─────────────────────────────────────
    # African Development Bank: official institutional news
    "African Development Bank":    "https://www.afdb.org/en/rss/news-and-events",
    # UN Africa: United Nations Africa news feed
    "UN Africa":                   "https://www.un.org/africarenewal/rss.xml",
}

class RSSConnector:
    """
    Fetch articles from one or more RSS / Atom feeds using feedparser.

    No API keys required.
    """

    SOURCE = "rss"

    def __init__(self):
        pass

    # Reuse NewsAPI domain credibility table for consistent scoring
    DOMAIN_CREDIBILITY: Dict[str, float] = {
        "reuters.com": 0.95, "apnews.com": 0.93, "bbc.co.uk": 0.92,
        "bbc.com": 0.92, "theguardian.com": 0.84, "npr.org": 0.88,
        "africacheck.org": 0.94, "dubawa.org": 0.92, "factcheckhub.com": 0.93,
        "pesacheck.org": 0.91, "factcheckafrica.net": 0.90,
        "premiumtimesng.com": 0.87, "thecable.ng": 0.85,
        "dailymaverick.co.za": 0.90, "groundup.org.za": 0.88,
        "punchng.com": 0.80, "channelstv.com": 0.82,
        "vanguardngr.com": 0.78, "myjoyonline.com": 0.78,
        "nation.africa": 0.81, "theeastafrican.co.ke": 0.82,
        "dailypost.ng": 0.75, "thenationonlineng.net": 0.74,
        "saharareporters.com": 0.70, "allafrica.com": 0.72,
        "informationng.com": 0.35, "naijaloaded.com.ng": 0.30,
    }

    def _domain_credibility(self, url: str) -> float:
        for domain, score in self.DOMAIN_CREDIBILITY.items():
            if domain in url:
                return score
        return 0.70   # RSS sources skew more reputable than unknown web pages

    def fetch(
        self,
        feeds: Optional[Dict[str, str]] = None,
        max_per_feed: int = 20,
        date_from: Optional[datetime] = None,
        date_to:   Optional[datetime] = None,
        keyword_filter: Optional[str] = None,
    ) -> List[NarrativePost]:
        """
        Fetch articles from RSS / Atom feeds.

        Parameters
        ----------
        feeds          : dict of {label: url}. Defaults to DEFAULT_RSS_FEEDS.
        max_per_feed   : maximum articles per feed
        date_from      : keep articles published on or after this UTC datetime.
                         Defaults to 48 h ago.
        date_to        : keep articles published on or before this UTC datetime.
                         Defaults to now.
        keyword_filter : if supplied (e.g. "Nigeria fuel subsidy"), only keep
                         entries whose title or summary contains at least one
                         of the keywords.
        """
        try:
            import feedparser  # noqa
        except ImportError:
            raise ImportError("Install feedparser: pip install feedparser")

        feeds  = feeds or DEFAULT_RSS_FEEDS
        now    = datetime.now(timezone.utc)
        cutoff_from = date_from or (now - timedelta(hours=48))
        cutoff_to   = date_to   or now
        posts: List[NarrativePost] = []

        # Build keyword list for relevance filtering
        filter_keywords: List[str] = []
        if keyword_filter:
            filter_keywords = [
                w.lower().strip()
                for w in keyword_filter.replace("-", " ").split()
                if len(w.strip()) > 2
            ]

        for feed_label, feed_url in feeds.items():
            try:
                parsed = feedparser.parse(feed_url)

                # Use the feed's own declared title as the outlet name,
                # falling back to the dict key only if unavailable.
                # e.g.  parsed.feed.title → "Premium Times"
                #       feed_label         → "Premium Times (Nigeria)"
                feed_outlet = (
                    getattr(parsed.feed, "title", None)
                    or feed_label
                ).strip()

                for entry in parsed.entries[:max_per_feed]:

                    # ── Publish date ──────────────────────────────────
                    pub = getattr(entry, "published_parsed", None)
                    if pub:
                        try:
                            created = datetime(*pub[:6], tzinfo=timezone.utc)
                        except Exception:
                            created = datetime.now(timezone.utc)
                    else:
                        created = datetime.now(timezone.utc)

                    if created < cutoff_from or created > cutoff_to:
                        continue

                    title   = (getattr(entry, "title",   "") or "").strip()
                    summary = (getattr(entry, "summary", "") or "").strip()
                    text    = " ".join(filter(None, [title, summary]))[:2000]

                    if not text.strip():
                        continue

                    # ── Keyword relevance filter ──────────────────────
                    # Only skip if we have actual filter words AND none
                    # of them appear in the entry text.
                    if filter_keywords:
                        text_lower = text.lower()
                        if not any(kw in text_lower for kw in filter_keywords):
                            continue

                    link = getattr(entry, "link", "") or ""

                    # ── Author: try entry fields before falling back ───
                    # Priority: entry.author → entry.author_detail.name
                    #           → entry.dc_creator → feed outlet name
                    entry_author = (
                        getattr(entry, "author", None)
                        or getattr(
                            getattr(entry, "author_detail", None),
                            "name", None)
                        or getattr(entry, "dc_creator", None)
                    )
                    author = (entry_author or feed_outlet).strip()

                    # ── Per-domain credibility ────────────────────────
                    cred = self._domain_credibility(link)

                    # ── Topic tags: entry categories first ───────────
                    entry_tags: List[str] = []
                    for tag_obj in getattr(entry, "tags", []):
                        label = (
                            getattr(tag_obj, "term",  None)
                            or getattr(tag_obj, "label", None)
                        )
                        if label:
                            entry_tags.append(label.lower().strip())
                    if not entry_tags:
                        # Fall back to meaningful words in the title
                        stop = {"the","and","for","with","that","this",
                                "from","have","been","will","are","was"}
                        entry_tags = [
                            w.lower().strip(".,;:'\"")
                            for w in title.split()
                            if len(w) > 4 and w.lower() not in stop
                        ][:6]
                    # Always stamp the feed region
                    entry_tags.append(
                        feed_label.lower().replace(" ", "_").replace("(","").replace(")","")
                    )

                    posts.append(NarrativePost(
                        post_id        = _stable_id(self.SOURCE,
                                                    link or text[:40]),
                        text           = text,
                        source         = self.SOURCE,
                        author         = author,
                        created_at     = created,
                        engagement     = 0,
                        reach_proxy    = 0,
                        url            = link,
                        topic_tags     = entry_tags,
                        credibility    = cred,
                        virality_score = round(1.0 - cred, 3),
                        raw            = {"feed_label": feed_label,
                                          "feed_outlet": feed_outlet},
                    ))

            except Exception as exc:
                logger.warning(f"RSS fetch failed for {feed_label}: {exc}")

        return posts


# ─────────────────────────────────────────────────────────────
# Demo mode  (synthetic, always works)
# ─────────────────────────────────────────────────────────────

_DEMO_TEMPLATES = [
    ("BREAKING: New study claims {topic} causes serious health risks, experts dismiss",
     0.15, 0.85),
    ("VIRAL: Government hiding truth about {topic} — leaked document reveals",
     0.10, 0.95),
    ("{topic} debunked by multiple independent fact-checkers",
     0.85, 0.10),
    ("Scientists publish peer-reviewed findings on {topic} in Nature",
     0.92, 0.20),
    ("Social media flooded with unverified claims about {topic}",
     0.60, 0.55),
    ("Fact-check: What we actually know about {topic}",
     0.88, 0.15),
    ("Conspiracy theory about {topic} spreads to millions before removal",
     0.20, 0.90),
    ("Official statement from WHO clarifies {topic} misinformation",
     0.90, 0.25),
    ("Anonymous source claims shocking revelation about {topic}",
     0.15, 0.80),
    ("University research team finds no evidence for {topic} claims",
     0.89, 0.20),
    ("Viral post misrepresents {topic} data — here's the real story",
     0.85, 0.35),
    ("{topic}: why experts say the viral narrative is dangerously wrong",
     0.80, 0.60),
]

_DEMO_AUTHORS = [
    "InfoWatch_Bot", "TruthSeeker99", "NewsFlash247", "FactBuster",
    "RealNews_Daily", "ViralAlert", "DebunkHub", "ScienceMatters",
    "BreakingAlerts", "MediaCritic", "WhistleblowerX", "CheckTheSource",
]


class DemoConnector:
    """
    Generates synthetic NarrativePost objects — no API keys needed.
    Useful for demonstrations, testing, and environments without internet.
    """

    SOURCE = "demo"

    def fetch(
        self,
        topic: str = "vaccines",
        n_posts: int = 150,
        seed: int = 42,
        fake_ratio: float = 0.35,
    ) -> List[NarrativePost]:
        """
        Parameters
        ----------
        topic      : keyword inserted into post templates
        n_posts    : number of synthetic posts to generate
        seed       : random seed for reproducibility
        fake_ratio : fraction of posts that are misinformation (credibility < 0.5)
        """
        rng   = np.random.default_rng(seed)
        posts = []
        now   = datetime.now(timezone.utc)

        for i in range(n_posts):
            tmpl, cred, viral = _DEMO_TEMPLATES[
                rng.integers(0, len(_DEMO_TEMPLATES))]
            text       = tmpl.format(topic=topic)
            author     = _DEMO_AUTHORS[rng.integers(0, len(_DEMO_AUTHORS))]
            minutes_ago = int(rng.integers(0, 48 * 60))
            engagement  = int(rng.integers(0, 50_000))
            reach       = int(rng.integers(100, 1_000_000))

            # Skew toward fake_ratio
            if rng.random() < fake_ratio:
                cred  = min(cred, 0.4)
                viral = max(viral, 0.5)
            else:
                cred  = max(cred, 0.5)

            posts.append(NarrativePost(
                post_id       = _stable_id(self.SOURCE, f"{i}_{seed}"),
                text          = text,
                source        = self.SOURCE,
                author        = author,
                created_at    = now - timedelta(minutes=minutes_ago),
                engagement    = engagement,
                reach_proxy   = reach,
                url           = f"https://demo.example.com/post/{i}",
                topic_tags    = [topic, f"tag_{rng.integers(1,10)}"],
                credibility   = float(np.clip(cred + rng.normal(0, 0.08), 0, 1)),
                virality_score= float(np.clip(viral + rng.normal(0, 0.06), 0, 1)),
                raw           = {},
            ))

        return posts


# ─────────────────────────────────────────────────────────────
# Unified DataFeed orchestrator
# ─────────────────────────────────────────────────────────────

class DataFeed:
    """
    Single entry-point that aggregates posts from all enabled sources,
    deduplicates, and converts to formats the simulation pipeline needs.

    Usage
    ─────
        feed = DataFeed(
            twitter_bearer="...",
            reddit_client_id="...", reddit_client_secret="...",
            newsapi_key="...",
        )
        posts = feed.fetch(
            query="vaccine misinformation",
            subreddits=["conspiracy", "skeptic"],
            rss_feeds=DEFAULT_RSS_FEEDS,
            max_per_source=100,
        )
        df   = feed.to_dataframe(posts)
        G    = feed.seed_graph(G, posts)
    """

    def __init__(
        self,
        twitter_bearer:     Optional[str] = None,
        reddit_client_id:   Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        newsapi_key:        Optional[str] = None,
        use_demo_fallback:  bool = True,
    ):
        self._sources: Dict[str, Any] = {}
        self._use_demo = use_demo_fallback

        if twitter_bearer or os.getenv("TWITTER_BEARER_TOKEN"):
            self._sources["twitter"] = TwitterConnector(twitter_bearer)

        if (reddit_client_id or os.getenv("REDDIT_CLIENT_ID")) and \
           (reddit_client_secret or os.getenv("REDDIT_CLIENT_SECRET")):
            self._sources["reddit"] = RedditConnector(
                reddit_client_id, reddit_client_secret)

        if newsapi_key or os.getenv("NEWSAPI_KEY"):
            self._sources["newsapi"] = NewsAPIConnector(newsapi_key)

        # RSS always available (no key)
        self._sources["rss"] = RSSConnector()

    @property
    def active_sources(self) -> List[str]:
        return list(self._sources.keys())

    def fetch(
        self,
        query:          str = "misinformation",
        subreddits:     Optional[List[str]] = None,
        rss_feeds:      Optional[Dict[str, str]] = None,
        max_per_source: int = 100,
        date_from:      Optional[datetime] = None,
        date_to:        Optional[datetime] = None,
        demo_topic:     str = "vaccines",
        demo_n:         int = 150,
        demo_seed:      int = 42,
    ) -> List[NarrativePost]:
        """
        Fetch from all configured sources and return a deduplicated list.

        Parameters
        ----------
        query          : search keyword(s) for Twitter, NewsAPI, RSS filter
        subreddits     : list of subreddit names (Reddit only)
        rss_feeds      : custom feed dict; defaults to DEFAULT_RSS_FEEDS
        max_per_source : max posts per source
        date_from      : start of date window (UTC datetime).
                         Defaults to 48 h ago when None.
        date_to        : end of date window (UTC datetime).
                         Defaults to now when None.
        demo_topic     : topic label for synthetic demo posts
        demo_n         : number of demo posts to generate
        demo_seed      : random seed for demo reproducibility
        """
        all_posts: List[NarrativePost] = []
        now       = datetime.now(timezone.utc)
        _from     = date_from or (now - timedelta(hours=48))
        _to       = date_to   or now

        # Twitter
        if "twitter" in self._sources:
            try:
                tw = self._sources["twitter"].fetch(
                    query=query, max_results=max_per_source,
                    date_from=_from, date_to=_to)
                all_posts.extend(tw)
                logger.info(f"Twitter: {len(tw)} posts")
            except Exception as exc:
                logger.warning(f"Twitter disabled: {exc}")

        # Reddit
        if "reddit" in self._sources:
            subs = subreddits or ["worldnews", "conspiracy", "skeptic",
                                   "politics", "science"]
            try:
                rd = self._sources["reddit"].fetch(
                    subreddits=subs,
                    limit=max_per_source // len(subs),
                    date_from=_from, date_to=_to)
                all_posts.extend(rd)
                logger.info(f"Reddit: {len(rd)} posts")
            except Exception as exc:
                logger.warning(f"Reddit disabled: {exc}")

        # NewsAPI
        if "newsapi" in self._sources:
            try:
                na = self._sources["newsapi"].fetch(
                    query=query, max_articles=max_per_source,
                    date_from=_from, date_to=_to)
                all_posts.extend(na)
                logger.info(f"NewsAPI: {len(na)} posts")
            except Exception as exc:
                logger.warning(f"NewsAPI disabled: {exc}")

        # RSS — always try, pass query as keyword filter
        if "rss" in self._sources:
            try:
                rss = self._sources["rss"].fetch(
                    feeds=rss_feeds, max_per_feed=20,
                    date_from=_from, date_to=_to,
                    keyword_filter=query)
                all_posts.extend(rss)
                logger.info(f"RSS: {len(rss)} posts")
            except Exception as exc:
                logger.warning(f"RSS disabled: {exc}")

        # Demo fallback
        if not all_posts and self._use_demo:
            logger.info("No real data — using demo mode")
            all_posts = DemoConnector().fetch(
                topic=demo_topic, n_posts=demo_n, seed=demo_seed)

        # Deduplicate by post_id
        seen, unique = set(), []
        for p in all_posts:
            if p.post_id not in seen:
                seen.add(p.post_id)
                unique.append(p)

        # Sort by virality descending
        unique.sort(key=lambda p: p.virality_score, reverse=True)
        logger.info(f"Total unique posts: {len(unique)}")
        return unique

    # ── Conversion helpers ────────────────────────────────

    def to_dataframe(self, posts: List[NarrativePost]) -> pd.DataFrame:
        """Convert posts to a DataFrame compatible with the ISOT pipeline."""
        rows = [p.to_dict() for p in posts]
        df   = pd.DataFrame(rows)

        # Add ISOT-compatible 'label' column:
        #   credibility < 0.5 → fake (1), else real (0)
        df['label'] = (df['credibility'] < 0.5).astype(int)
        df['text']  = df['text'].fillna("").astype(str)
        return df

    def seed_graph(
        self,
        G,
        posts: List[NarrativePost],
        virality_threshold: float = 0.65,
    ) -> tuple:
        """
        Assign posts to graph nodes and identify high-virality seed nodes.

        Returns
        ───────
        G         : graph with 'article_text', 'article_label',
                    'virality_score', 'source' attributes per node
        seeds     : list of node IDs with virality ≥ threshold
        """
        import networkx as nx
        nodes = list(G.nodes())
        n     = len(nodes)

        if not posts:
            return G, []

        # Cycle through posts if fewer than nodes
        extended = (posts * (n // len(posts) + 1))[:n]

        attrs = {}
        for node, post in zip(nodes, extended):
            attrs[node] = {
                'article_text':    post.text,
                'article_label':   1 if post.credibility < 0.5 else 0,
                'virality_score':  post.virality_score,
                'source':          post.source,
                'post_id':         post.post_id,
                'credibility':     post.credibility,
            }
        nx.set_node_attributes(G, attrs)

        # Seed nodes: those assigned the most viral posts
        seeds = [node for node, post in zip(nodes, extended)
                 if post.virality_score >= virality_threshold]

        # Ensure at least 1 seed
        if not seeds:
            seeds = [nodes[0]]

        return G, seeds

    def narrative_summary(self, posts: List[NarrativePost]) -> Dict:
        """
        Return a summary dict for dashboard display.
        """
        if not posts:
            return {}

        df = pd.DataFrame([p.to_dict() for p in posts])
        df['credibility']    = df['credibility'].astype(float)
        df['virality_score'] = df['virality_score'].astype(float)

        return {
            'total_posts':      len(posts),
            'sources':          df['source'].value_counts().to_dict(),
            'avg_virality':     round(df['virality_score'].mean(), 3),
            'avg_credibility':  round(df['credibility'].mean(), 3),
            'fake_count':       int((df['credibility'] < 0.5).sum()),
            'real_count':       int((df['credibility'] >= 0.5).sum()),
            'top_authors':      df.nlargest(5, 'virality_score')[
                                    ['author','virality_score','source']
                                ].to_dict('records'),
            'top_tags':         _top_tags(posts),
            'time_range': {
                'earliest': df['created_at'].min(),
                'latest':   df['created_at'].max(),
            },
        }


def _top_tags(posts: List[NarrativePost], top_n: int = 10) -> List[str]:
    from collections import Counter
    all_tags = [tag for p in posts for tag in p.topic_tags]
    return [tag for tag, _ in Counter(all_tags).most_common(top_n)]