import asyncio
import json
import argparse
import sys

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
import mcp.server.sse

# Import helper functions for Yahoo Finance functionality
from .helper import (
    get_ticker_info,
    get_ticker_news,
    search_yahoo_finance,
    get_top_entities,
    get_price_history,
    get_ticker_option_chain,
    get_ticker_earnings,
    get_ticker_filings,
    get_filing_content,
)

# Initialize the MCP server
server = Server("yahoo_finance_server")


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available resources.
    Currently no resources are exposed by this server.
    """
    return []


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a specific resource by its URI.
    Currently no resources are supported.
    """
    raise ValueError(f"Unsupported resource URI: {uri}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Currently no prompts are exposed by this server.
    """
    return []


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by name.
    Currently no prompts are supported.
    """
    raise ValueError(f"Unknown prompt: {name}")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available Yahoo Finance tools.
    """
    return [
        types.Tool(
            name="get-ticker-info",
            description="Retrieve comprehensive stock data including company info, financials, trading metrics and governance data",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')",
                    }
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-ticker-news",
            description="Fetch recent news articles related to a specific stock symbol with title, content, and source details",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol to get news for",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of news articles to fetch (default: 10, maximum: 50)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="search",
            description="Search Yahoo Finance for stocks, ETFs, and other financial instruments",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (company name, ticker symbol, etc.)",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of search results to return (default: 10, maximum: 25)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 25,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get-top-entities",
            description="Get top entities (ETFs, mutual funds, companies, growth companies, or performing companies) in a sector",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "etfs",
                            "mutual_funds",
                            "companies",
                            "growth_companies",
                            "performing_companies",
                        ],
                        "description": "Type of entities to retrieve",
                    },
                    "sector": {
                        "type": "string",
                        "description": "Sector name (technology, healthcare, financial, energy, consumer, industrial)",
                        "default": "",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of entities to return (default: 10, maximum: 20)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["entity_type"],
            },
        ),
        types.Tool(
            name="get-price-history",
            description="Fetch historical price data for a given stock symbol over a specified period and interval",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "period": {
                        "type": "string",
                        "enum": [
                            "1d",
                            "5d",
                            "1mo",
                            "3mo",
                            "6mo",
                            "1y",
                            "2y",
                            "5y",
                            "10y",
                            "ytd",
                            "max",
                        ],
                        "description": "Period to fetch data for",
                        "default": "1y",
                    },
                    "interval": {
                        "type": "string",
                        "enum": [
                            "1m",
                            "2m",
                            "5m",
                            "15m",
                            "30m",
                            "60m",
                            "90m",
                            "1h",
                            "1d",
                            "5d",
                            "1wk",
                            "1mo",
                            "3mo",
                        ],
                        "description": "Data interval",
                        "default": "1d",
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="ticker-option-chain",
            description="Get most recent or around certain date option chain data. Parameters include call or put, and date. If no date, use most recent top 10 day forward dates",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "option_type": {
                        "type": "string",
                        "enum": ["call", "put", "both"],
                        "description": "Type of options to retrieve",
                        "default": "both",
                    },
                    "date": {
                        "type": "string",
                        "description": "Specific expiration date in YYYY-MM-DD format. If not provided, uses most recent available dates",
                        "default": None,
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="ticker-earning",
            description="Get earnings data including annual or quarterly data, and upcoming earnings dates. Parameters include annual or quarter, and date. If no date, use most recent, also include the date of upcoming earning time if available",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["annual", "quarterly"],
                        "description": "Earnings period to retrieve",
                        "default": "annual",
                    },
                    "date": {
                        "type": "string",
                        "description": "Specific date in YYYY-MM-DD format. If not provided, uses most recent data",
                        "default": None,
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-sec-filings",
            description="Retrieve recent SEC filings for a stock symbol including 10-K, 10-Q, 8-K and other regulatory filings with document details and links",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol to get SEC filings for",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of SEC filings to fetch (default: 100)",
                        "default": 100,
                        "minimum": 1,
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-filing-content",
            description="Download and retrieve the full content of a specific SEC filing document by URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the SEC filing document to download (obtained from get-sec-filings tool)",
                    },
                },
                "required": ["url"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle Yahoo Finance tool execution requests.
    """
    if name == "get-ticker-info":
        return await _handle_get_ticker_info(arguments)
    elif name == "get-ticker-news":
        return await _handle_get_ticker_news(arguments)
    elif name == "search":
        return await _handle_search(arguments)
    elif name == "get-top-entities":
        return await _handle_get_top_entities(arguments)
    elif name == "get-price-history":
        return await _handle_get_price_history(arguments)
    elif name == "ticker-option-chain":
        return await _handle_ticker_option_chain(arguments)
    elif name == "ticker-earning":
        return await _handle_ticker_earning(arguments)
    elif name == "get-sec-filings":
        return await _handle_get_sec_filings(arguments)
    elif name == "get-filing-content":
        return await _handle_get_filing_content(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _handle_get_ticker_info(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-ticker-info tool execution using only fast_info fields.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for ticker info retrieval")

    try:
        symbol = arguments["symbol"].upper()
        ticker_info = await get_ticker_info(symbol)

        # ticker_info is already a JSON string from helper.py
        return [
            types.TextContent(
                type="text",
                text=ticker_info,
            )
        ]

    except Exception as e:
        error_response = json.dumps(
            {"symbol": arguments.get("symbol", "unknown"), "error": str(e)}
        )
        return [
            types.TextContent(
                type="text",
                text=error_response,
            )
        ]


async def _handle_get_ticker_news(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-ticker-news tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for news retrieval")

    try:
        symbol = arguments["symbol"].upper()
        count = arguments.get("count", 10)

        news_data = await get_ticker_news(symbol, count)

        if not news_data.get("news"):
            return [
                types.TextContent(
                    type="text",
                    text=f"📰 No news found for {symbol}",
                )
            ]

        # Format the response nicely
        news_text = f"""📰 **Recent News for {symbol}** ({news_data['news_count']} articles)

"""

        for i, article in enumerate(news_data["news"], 1):
            summary_text = (
                f"\n• **Summary:** {article['summary']}"
                if article.get("summary") and article["summary"].strip()
                else ""
            )
            news_text += f"""**{i}. {article['title']}**
• **Publisher:** {article['publisher']}
• **Published:** {article['published']}
• **Link:** {article['link']}{summary_text}

"""

        return [
            types.TextContent(
                type="text",
                text=news_text,
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving news for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


async def _handle_search(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle search tool execution.
    """
    if not arguments or not arguments.get("query"):
        raise ValueError("Query is required for search")

    try:
        query = arguments["query"]
        count = arguments.get("count", 10)

        search_data = await search_yahoo_finance(query, count)

        if not search_data.get("results") and not search_data.get("news"):
            return [
                types.TextContent(
                    type="text",
                    text=f"🔍 No results found for '{query}'",
                )
            ]

        return [
            types.TextContent(
                type="text",
                text=json.dumps(search_data),
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error searching for '{arguments.get('query', 'unknown')}': {str(e)}",
            )
        ]


async def _handle_get_top_entities(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-top-entities tool execution.
    """
    if not arguments or not arguments.get("entity_type"):
        raise ValueError("Entity type is required")

    try:
        entity_type = arguments["entity_type"]
        sector = arguments.get("sector", "")
        count = arguments.get("count", 10)

        entities_data = await get_top_entities(entity_type, sector, count)

        if not entities_data.get("results"):
            return [
                types.TextContent(
                    type="text",
                    text=f"🏆 No {entity_type} found for sector '{sector}'",
                )
            ]

        return [
            types.TextContent(
                type="text",
                text=json.dumps(entities_data),
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving top entities: {str(e)}",
            )
        ]


async def _handle_get_price_history(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-price-history tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for price history")

    try:
        symbol = arguments["symbol"].upper()
        period = arguments.get("period", "1y")
        interval = arguments.get("interval", "1d")

        history_data = await get_price_history(symbol, period, interval)

        if not history_data.get("data"):
            return [
                types.TextContent(
                    type="text",
                    text=f"📈 No price history found for {symbol}",
                )
            ]

        # Format the response nicely - show last 10 data points
        history_text = f"""📈 **Price History for {symbol}**
**Period:** {period} | **Interval:** {interval} | **Data Points:** {history_data['count']}

**Recent Data (Last 10 points):**
"""

        recent_data = history_data["data"][-10:]  # Get last 10 data points
        for data_point in recent_data:
            history_text += f"""• **{data_point['date'][:10]}**: Open: ${data_point['open']:.2f}, High: ${data_point['high']:.2f}, Low: ${data_point['low']:.2f}, Close: ${data_point['close']:.2f}, Volume: {data_point['volume']:,}
"""

        # Add summary statistics
        closes = [d["close"] for d in history_data["data"] if d["close"] is not None]
        if closes:
            min_price = min(closes)
            max_price = max(closes)
            avg_price = sum(closes) / len(closes)

            history_text += f"""
**Summary Statistics:**
• **Period Low:** ${min_price:.2f}
• **Period High:** ${max_price:.2f}
• **Average Price:** ${avg_price:.2f}
• **Total Change:** {((closes[-1] - closes[0]) / closes[0] * 100):.2f}%
"""

        return [
            types.TextContent(
                type="text",
                text=history_text,
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving price history for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


async def _handle_ticker_option_chain(
    arguments: dict | None,
) -> list[types.TextContent]:
    """
    Handle ticker-option-chain tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for option chain")

    try:
        symbol = arguments["symbol"].upper()
        option_type = arguments.get("option_type", "both")
        date = arguments.get("date")

        options_data = await get_ticker_option_chain(symbol, option_type, date)

        if options_data.get("error"):
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ {options_data['error']}",
                )
            ]

        # Format the response nicely
        options_text = f"""⚡ **Option Chain for {symbol}**
**Expiration Date:** {options_data.get('expiration_date', 'N/A')}
**Available Expirations:** {', '.join(options_data.get('available_expirations', [])[:5])}

"""

        if option_type in ["call", "both"] and "calls" in options_data:
            options_text += "**📈 CALL Options:**\n"
            calls = options_data["calls"][:10]  # Show first 10
            for call in calls:
                options_text += f"""• Strike: ${call['strike']:.2f} | Last: ${call['last_price']:.2f} | Bid: ${call['bid']:.2f} | Ask: ${call['ask']:.2f} | Vol: {call['volume']} | OI: {call['open_interest']} | IV: {call['implied_volatility']:.2%}
"""
            options_text += "\n"

        if option_type in ["put", "both"] and "puts" in options_data:
            options_text += "**📉 PUT Options:**\n"
            puts = options_data["puts"][:10]  # Show first 10
            for put in puts:
                options_text += f"""• Strike: ${put['strike']:.2f} | Last: ${put['last_price']:.2f} | Bid: ${put['bid']:.2f} | Ask: ${put['ask']:.2f} | Vol: {put['volume']} | OI: {put['open_interest']} | IV: {put['implied_volatility']:.2%}
"""

        return [
            types.TextContent(
                type="text",
                text=options_text,
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving option chain for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


async def _handle_ticker_earning(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle ticker-earning tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for earnings data")

    try:
        symbol = arguments["symbol"].upper()
        period = arguments.get("period", "annual")
        date = arguments.get("date")

        earnings_data = await get_ticker_earnings(symbol, period, date)

        if earnings_data.get("error"):
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Error retrieving earnings: {earnings_data['error']}",
                )
            ]

        # Format the response nicely
        earnings_text = f"""💰 **Earnings Data for {symbol}**
**Period:** {period.title()}

**Key Metrics:**
• **Trailing EPS:** ${earnings_data.get('trailing_eps', 0):.2f}
• **Forward EPS:** ${earnings_data.get('forward_eps', 0):.2f}
• **P/E Ratio:** {earnings_data.get('pe_ratio', 0):.2f}
• **Forward P/E:** {earnings_data.get('forward_pe', 0):.2f}

"""

        # Historical earnings data
        if earnings_data.get("earnings_data"):
            earnings_text += f"**Historical {period.title()} Earnings:**\n"
            recent_earnings = earnings_data["earnings_data"][:4]  # Show last 4
            for earning in recent_earnings:
                earnings_text += f"""• **{earning['date']}**: Revenue: ${earning['total_revenue']:,} | Net Income: ${earning['net_income']:,} | EBITDA: ${earning['ebitda']:,}
"""
            earnings_text += "\n"

        # Upcoming earnings
        if earnings_data.get("upcoming_earnings"):
            earnings_text += "**Upcoming Earnings:**\n"
            for upcoming in earnings_data["upcoming_earnings"]:
                earnings_text += f"""• **Date:** {upcoming['earnings_date']} | **EPS Est:** ${upcoming['eps_estimate']:.2f}
"""

        return [
            types.TextContent(
                type="text",
                text=earnings_text,
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving earnings for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


async def _handle_get_sec_filings(arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle get-sec-filings tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for SEC filings retrieval")

    try:
        symbol = arguments["symbol"].upper()
        count = arguments.get("count", 10)

        filings_data = await get_ticker_filings(symbol, count)

        if filings_data.get("error"):
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Error retrieving SEC filings: {filings_data['error']}",
                )
            ]

        if not filings_data.get("filings"):
            return [
                types.TextContent(
                    type="text",
                    text=f"📋 No SEC filings found for {symbol}",
                )
            ]

        # Format the response nicely
        filings_text = f"""📋 **SEC Filings for {symbol}** ({filings_data['filings_count']} filings)

"""

        for i, filing in enumerate(filings_data["filings"], 1):
            filing_date = filing.get("date") or filing.get("filing_date", "")
            acceptance_date = filing.get("acceptance_date", "")
            report_date = filing.get("report_date", "")
            edgar_url = filing.get("edgar_url", "")
            
            date_info = f"📅 **Date:** {filing_date}"
            if acceptance_date:
                date_info += f" | **Accepted:** {acceptance_date}"
            if report_date:
                date_info += f" | **Report Date:** {report_date}"
            
            # Handle exhibits - show all available documents (URLs only, no content)
            exhibits_info = ""
            if filing.get("exhibits") and len(filing["exhibits"]) > 0:
                exhibits_info = f"\n📄 **Documents ({filing.get('total_exhibits', 0)}):**"
                for exhibit in filing["exhibits"]:
                    exhibits_info += f"\n  • **{exhibit['exhibit_type']}**: {exhibit['url']}"
            elif edgar_url:
                exhibits_info = f"\n🔗 **EDGAR Link:** {edgar_url}"
            elif filing.get("url"):
                exhibits_info = f"\n🔗 **Document Link:** {filing['url']}"
                
            # Add note about content retrieval
            if filing.get("exhibits") and len(filing["exhibits"]) > 0:
                exhibits_info += f"\n💡 **Note:** Use get-filing-content tool with URLs above to retrieve document content"

            filings_text += f"""**{i}. {filing['type']}**
📝 **Title:** {filing.get('title', 'N/A')}
{date_info}{exhibits_info}

"""

        # Return the main summary as text, but we could extend this to return
        # individual documents as EmbeddedResource for binary content
        results = [
            types.TextContent(
                type="text",
                text=filings_text,
            )
        ]
        
        # Optionally, we could add individual documents as embedded resources
        # for direct access to file content (especially for PDFs, images, etc.)
        # This would allow tools to access the raw file content directly
        for filing in filings_data["filings"]:
            for exhibit in filing.get("exhibits", []):
                if exhibit.get("content") and not exhibit.get("error"):
                    # For binary content (PDFs, etc.), we could add as EmbeddedResource
                    if exhibit.get("content_type") in ["application/pdf", "image/png", "image/jpeg"]:
                        # Note: This would require the content to be properly formatted
                        # For now, keeping as text summary, but structure is ready for expansion
                        pass
        
        return results

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving SEC filings for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


async def _handle_get_filing_content(arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle get-filing-content tool execution.
    """
    if not arguments or not arguments.get("url"):
        raise ValueError("URL is required for filing content retrieval")

    try:
        url = arguments["url"]
        
        content_data = await get_filing_content(url)
        
        if content_data.get("error"):
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Error retrieving filing content: {content_data['error']}",
                )
            ]
        
        # Format the response with content info and preview
        content_text = f"""📄 **SEC Filing Content**
🔗 **URL:** {content_data['url']}
📊 **Content Type:** {content_data['content_type']}
📏 **Size:** {content_data['size']} bytes
✅ **Status:** {content_data['status']}

"""
        
        # Add content based on type and size
        if content_data.get("content"):
            content = content_data["content"]
            
            if content_data["content_type"].startswith("text/") or "xml" in content_data["content_type"]:
                # For text content, show the full content
                content_text += f"""**📋 Content:**
```
{content}
```"""
            elif content_data["content_type"] == "application/pdf":
                # For PDF, show that it's base64 encoded
                content_text += f"""**📋 Content:** PDF document (base64 encoded)
**💡 Note:** This is a PDF file encoded in base64 format. The content can be decoded and saved as a PDF file.

**🔍 Base64 Content Preview:**
```
{content[:500]}...
```"""
            else:
                # For other binary content
                content_text += f"""**📋 Content:** Binary file (base64 encoded)
**💡 Note:** This is a binary file encoded in base64 format.

**🔍 Base64 Content Preview:**
```
{content[:500]}...
```"""
        
        return [
            types.TextContent(
                type="text",
                text=content_text,
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving filing content for {arguments.get('url', 'unknown')}: {str(e)}",
            )
        ]


async def main():
    """Main entry point for the Yahoo Finance MCP server."""
    parser = argparse.ArgumentParser(description="Yahoo Finance MCP Server")
    parser.add_argument(
        "--transport", 
        choices=["stdio", "http"], 
        default="stdio",
        help="Transport method (stdio or http)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port for HTTP server (default: 3000)"
    )
    
    args = parser.parse_args()
    
    init_options = InitializationOptions(
        server_name="yahoo-finance-server",
        server_version="0.1.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )
    
    if args.transport == "http":
        print(f"Starting Yahoo Finance MCP Server with HTTP transport on http://{args.host}:{args.port}")
        print(f"Inspector URL: http://{args.host}:{args.port}")
        
        try:
            # Use FastMCP for HTTP/StreamableHttp setup
            from mcp.server.fastmcp import FastMCP
            import uvicorn
            
            # Create FastMCP server with HTTP transport (replaces deprecated SSE)
            fastmcp_server = FastMCP(
                name="yahoo-finance-server",
                host=args.host,
                port=args.port,
                # Use streamable_http instead of deprecated sse
                streamable_http_path="/",
                json_response=False  # Use MCP format instead of JSON
            )
            
            # Register all our tools with FastMCP
            @fastmcp_server.tool()
            async def get_ticker_info(symbol: str) -> str:
                """Get comprehensive stock information"""
                from .helper import get_ticker_info as helper_get_ticker_info
                return await helper_get_ticker_info(symbol)
                
            @fastmcp_server.tool()
            async def get_ticker_news(symbol: str, count: int = 10) -> dict:
                """Get recent news for a stock symbol"""
                from .helper import get_ticker_news as helper_get_ticker_news
                return await helper_get_ticker_news(symbol, count)
                
            @fastmcp_server.tool()
            async def search(query: str, count: int = 10) -> dict:
                """Search Yahoo Finance for stocks and instruments"""
                from .helper import search_yahoo_finance
                return await search_yahoo_finance(query, count)
                
            @fastmcp_server.tool()
            async def get_top_entities(entity_type: str, sector: str = "", count: int = 10) -> dict:
                """Get top entities in a sector"""
                from .helper import get_top_entities as helper_get_top_entities
                return await helper_get_top_entities(entity_type, sector, count)
                
            @fastmcp_server.tool()
            async def get_price_history(symbol: str, period: str = "1y", interval: str = "1d") -> dict:
                """Get historical price data"""
                from .helper import get_price_history as helper_get_price_history
                return await helper_get_price_history(symbol, period, interval)
                
            @fastmcp_server.tool()
            async def ticker_option_chain(symbol: str, option_type: str = "both", date: str = None) -> dict:
                """Get option chain data"""
                from .helper import get_ticker_option_chain
                return await get_ticker_option_chain(symbol, option_type, date)
                
            @fastmcp_server.tool()
            async def ticker_earning(symbol: str, period: str = "annual", date: str = None) -> dict:
                """Get earnings data"""
                from .helper import get_ticker_earnings
                return await get_ticker_earnings(symbol, period, date)
                
            @fastmcp_server.tool()
            async def get_sec_filings(symbol: str, count: int = 100) -> dict:
                """Get SEC filings for a stock symbol"""
                from .helper import get_ticker_filings
                return await get_ticker_filings(symbol, count)
            
            @fastmcp_server.tool()
            async def get_filing_content(url: str) -> dict:
                """Download and retrieve the full content of a specific SEC filing document by URL"""
                from .helper import get_filing_content as helper_get_filing_content
                return await helper_get_filing_content(url)
            
            # Get the ASGI app from FastMCP
            app = fastmcp_server.streamable_http_app()
            
            # Run the server using async approach
            config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
            server_instance = uvicorn.Server(config)
            await server_instance.serve()
            
        except ImportError as e:
            print(f"❌ HTTP transport dependencies not available: {e}")
            print("Install required dependencies: pip install uvicorn starlette")
            print("Or use stdio transport: python -m src.yahoo_finance_server --transport stdio")
            return
    else:
        print("Starting Yahoo Finance MCP Server with stdio transport")
        # Use stdio transport for MCP communication
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)
