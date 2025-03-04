import express from 'express';
import { ApiPromise, WsProvider } from '@polkadot/api';

const router = express.Router();

// Cache mechanism to avoid too frequent requests
let priceCache = {
    price: null,
    timestamp: 0,
    ttl: 5 * 60 * 1000 // 5 minutes cache TTL
};

/**
 * GET /api/price
 * Returns the current token price for a subnet
 * Query Parameters:
 * - subnetId (optional): defaults to 3
 */
router.get('/', async (req, res, next) => {
    try {
        const subnetId = parseInt(req.query.subnetId || '3', 10);
        const now = Date.now();

        // Return cached price if available and not expired
        if (priceCache.price && (now - priceCache.timestamp) < priceCache.ttl) {
            return res.json({
                success: true,
                data: {
                    price: priceCache.price,
                    subnetId: subnetId,
                    cached: true,
                    timestamp: new Date(priceCache.timestamp).toISOString()
                },
                timestamp: new Date().toISOString()
            });
        }

        console.log("Connecting to Polkadot API...");
        // Use environment variable with fallback
        const networkUrl = process.env.NETWORK_URL || "ws://162.55.159.123:11144";
        const provider = new WsProvider(networkUrl);

        const api = await ApiPromise.create({ provider });
        await api.isReady;
        console.log("API is ready");

        const namespace = "SubnetInfoRuntimeApi";
        const method = "get_all_dynamic_info";

        const result = await api.rpc.state.call(`${namespace}_${method}`, "0x");
        const typedef = api.registry.createLookupType(
            api.runtimeMetadata.asV15.apis
                .find(a => a.name.toHuman() === namespace)
                ?.methods.find(m => m.name.toHuman() === method)
                ?.output
        );
        const subnetsDynamicInfo = api.createType(typedef, result).toJSON();

        const subnetInfo = subnetsDynamicInfo.find(subnet => subnet.netuid === subnetId);
        if (!subnetInfo) {
            await provider.disconnect();
            return res.status(404).json({
                success: false,
                error: `Subnet ${subnetId} not found`,
                timestamp: new Date().toISOString()
            });
        }

        const price = subnetInfo.netuid === 0 ? 1 : subnetInfo.taoIn / subnetInfo.alphaIn;

        // Update cache
        priceCache = {
            price,
            timestamp: now,
            ttl: priceCache.ttl
        };

        await provider.disconnect();

        res.json({
            success: true,
            data: {
                price,
                subnetId,
                taoIn: subnetInfo.taoIn,
                alphaIn: subnetInfo.alphaIn,
                cached: false,
                timestamp: new Date().toISOString()
            },
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Error fetching price:', error);
        res.status(500).json({
            success: false,
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

export default router;