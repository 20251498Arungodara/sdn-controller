'use strict';

const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const grpc = require('@grpc/grpc-js');
const { connect, signers } = require('@hyperledger/fabric-gateway');

const app = express();
app.use(bodyParser.json());

const PORT = 3000;

/* =========================================================
   METRICS (GLOBAL)
========================================================= */

let epochCount = 0;              // throughput
const startTime = Date.now();    // experiment start

/* =========================================================
   Fabric helpers
========================================================= */

// gRPC connection to peer0.org1
function newGrpcConnection() {
    const tlsCertPath = path.resolve(
        __dirname,
        '../../test-network/organizations/peerOrganizations/org1.example.com',
        'peers/peer0.org1.example.com/tls/ca.crt'
    );

    const tlsRootCert = fs.readFileSync(tlsCertPath);
    const tlsCredentials = grpc.credentials.createSsl(tlsRootCert);

    return new grpc.Client(
        'localhost:7051',
        tlsCredentials,
        {
            'grpc.ssl_target_name_override': 'peer0.org1.example.com',
            'grpc.default_authority': 'peer0.org1.example.com'
        }
    );
}

// 🔐 MSP ADMIN identity
function newIdentity() {
    return {
        mspId: 'Org1MSP',
        credentials: fs.readFileSync(
            path.join(__dirname, 'wallet/Admin/cert.pem')
        )
    };
}

// 🔐 MSP ADMIN signer
function newSigner() {
    const privateKeyPem = fs.readFileSync(
        path.join(__dirname, 'wallet/Admin/key.pem')
    );
    const privateKey = crypto.createPrivateKey(privateKeyPem);
    return signers.newPrivateKeySigner(privateKey);
}

// Gateway + contract
async function getContract() {
    const grpcClient = newGrpcConnection();

    const gateway = connect({
        client: grpcClient,
        identity: newIdentity(),
        signer: newSigner(),

        // IMPORTANT: deterministic behavior
        discovery: false,

        evaluateOptions: () => ({ deadline: Date.now() + 5000 }),
        endorseOptions: () => ({ deadline: Date.now() + 15000 }),
        submitOptions: () => ({ deadline: Date.now() + 5000 }),
        commitStatusOptions: () => ({ deadline: Date.now() + 60000 }),
    });

    const network = gateway.getNetwork('mychannel');
    const contract = network.getContract('epochcontract');

    return { gateway, grpcClient, contract };
}



/* =========================================================
   API ROUTES
========================================================= */

// Register SDN controller
app.post('/registerController', async (req, res) => {
    try {
        const { controller_id, pubkey_pem } = req.body;

        const { gateway, grpcClient, contract } = await getContract();

        const result = await contract.submitTransaction(
            'RegisterController',
            controller_id,
            pubkey_pem
        );

        gateway.close();
        grpcClient.close();

        res.json({
            status: result.toString()
        });
    } catch (e) {
        console.error('RegisterController error:', e);
        res.status(500).json({ error: e.message });
    }
});

// Submit epoch data + METRICS
app.post('/submitEpoch', async (req, res) => {
    try {
        const { gateway, grpcClient, contract } = await getContract();

        const tStart = Date.now();   // ⏱ latency start

        const result = await contract.submitTransaction(
            'SubmitEpoch',
            JSON.stringify(req.body)
        );

        const latency = Date.now() - tStart; // ⏱ latency end
        epochCount++;

        // ---- CSV logging (paper metrics) ----
        const logLine = `${req.body.epoch},${latency},${Date.now()}\n`;
        fs.appendFileSync('metrics.csv', logLine);

        gateway.close();
        grpcClient.close();

        res.json({
            status: result.toString(),
            latency_ms: latency,
            epochs_submitted: epochCount
        });

        console.log(
            `📊 Epoch ${req.body.epoch} committed | latency=${latency} ms`
        );
    } catch (e) {
        console.error('SubmitEpoch error:', e);
        res.status(500).json({ error: e.message });
    }
});


/* =========================================================
   QUERY ROUTES (READ-ONLY)
========================================================= */

// Query controller
app.get('/queryController/:controllerId', async (req, res) => {
    try {
        const { gateway, grpcClient, contract } = await getContract();

        const result = await contract.evaluateTransaction(
            'QueryController',
            req.params.controllerId
        );

        gateway.close();
        grpcClient.close();

        res.json(JSON.parse(result.toString()));
    } catch (e) {
        console.error('QueryController error:', e);
        res.status(404).json({ error: e.message });
    }
});

// Query epoch
app.get('/queryEpoch/:epoch/:controllerId', async (req, res) => {
    try {
        const { gateway, grpcClient, contract } = await getContract();

        const result = await contract.evaluateTransaction(
            'QueryEpoch',
            req.params.epoch,
            req.params.controllerId
        );

        gateway.close();
        grpcClient.close();

        res.json(JSON.parse(result.toString()));
    } catch (e) {
        console.error('QueryEpoch error:', e);
        res.status(404).json({ error: e.message });
    }
});

// Metrics endpoint (FOR PAPER)
app.get('/metrics', (req, res) => {
    const elapsedSeconds = (Date.now() - startTime) / 1000;
    const throughput = epochCount / elapsedSeconds;

    res.json({
        epochs_submitted: epochCount,
        elapsed_seconds: elapsedSeconds.toFixed(2),
        throughput_eps: throughput.toFixed(3)
    });
});

/* ========================================================= */

app.listen(PORT, () => {
    console.log(`🚀 Gateway API running at http://localhost:${PORT}`);
});
