const { Gateway, Wallets } = require("fabric-network");
const fs = require("fs");
const path = require("path");

const ccpPath = path.resolve(__dirname, "connection-org1.json");
const walletPath = path.join(__dirname, "wallet");

const CHANNEL = "mychannel";
const CC = "epochcontract";
const USER = "appUser";

async function getContract() {
  const ccp = JSON.parse(fs.readFileSync(ccpPath));
  const wallet = await Wallets.newFileSystemWallet(walletPath);

  const gateway = new Gateway();
  await gateway.connect(ccp, {
    wallet,
    identity: USER,
    discovery: { enabled: true, asLocalhost: true }
  });

  const network = await gateway.getNetwork(CHANNEL);
  return { contract: network.getContract(CC), gateway };
}

async function registerController(id, pubkey) {
  const { contract, gateway } = await getContract();
  try {
    const res = await contract.submitTransaction("RegisterController", id, pubkey);
    return JSON.parse(res.toString());
  } finally {
    gateway.disconnect();
  }
}

async function submitEpoch(epochJSON) {
  const { contract, gateway } = await getContract();
  try {
    return await contract.submitTransaction("SubmitEpoch", epochJSON);
  } finally {
    gateway.disconnect();
  }
}

async function queryEpoch(id) {
  const { contract, gateway } = await getContract();
  try {
    return await contract.evaluateTransaction("QueryEpoch", id);
  } finally {
    gateway.disconnect();
  }
}

module.exports = { registerController, submitEpoch, queryEpoch };
