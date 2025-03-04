import { ApiPromise, WsProvider } from '@polkadot/api';

const connectApi = async () => {
    console.log("Connecting to Polkadot API...");

    // Create provider
    const provider = new WsProvider("ws://162.55.159.123:11144");

    // Create API with explicit wait for ready
    const api = await ApiPromise.create({
        provider: provider,
    });

    // Wait for API to be ready
    await api.isReady;
    console.log("API is ready");

    const subnetId = 3;
    const namespace = "SubnetInfoRuntimeApi";
    const method = "get_all_dynamic_info";

    try {
        // Verify that api.rpc.state exists
        if (!api.rpc || !api.rpc.state) {
            console.error("API.rpc.state is not available. API connection may not be complete.");
            return;
        }

        const result = await api.rpc.state.call(`${namespace}_${method}`, "0x");
        const typedef = api.registry.createLookupType(
            api.runtimeMetadata.asV15.apis
                .find(a => a.name.toHuman() === namespace)
                ?.methods.find(m => m.name.toHuman() === method)
                ?.output
        );
        const subnetsDynamicInfo = api.createType(typedef, result).toJSON();

        const subnetInfo = subnetsDynamicInfo.find(subnet => subnet.netuid === subnetId);
        const price = subnetInfo.netuid === 0 ? 1 : subnetInfo.taoIn / subnetInfo.alphaIn;

        console.log(`Subnet ${subnetId} price:`, price);
    } catch (error) {
        console.error('Error:', error);
    } finally {
        // Disconnect when done
        await provider.disconnect();
    }
};

// Add proper error handling for the main function
connectApi().catch(err => {
    console.error("Fatal error:", err);
    process.exit(1);
});