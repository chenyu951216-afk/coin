from __future__ import annotations

import os

from coinlab.market_research_package import install_strict_market_packager

# Install the development-only packager before the v0.7 server imports the
# market backtest module. Research ZIPs then exclude both embargo gaps and the
# locked test holdout; the full audit ZIP keeps them for final promotion only.
install_strict_market_packager()

from coinlab.server_v7 import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
