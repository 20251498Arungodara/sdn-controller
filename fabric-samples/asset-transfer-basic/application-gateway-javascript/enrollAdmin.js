'use strict';

const FabricCAServices = require('fabric-ca-client');
const fs = require('fs');
const path = require('path');

async function main() {
    const caURL = 'https://localhost:7054';
    const ca = new FabricCAServices(caURL);

    const walletPath = path.join(__dirname, 'wallet');
    if (!fs.existsSync(walletPath)) {
        fs.mkdirSync(walletPath);
    }

    const certPath = path.join(walletPath, 'admin-cert.pem');
    const keyPath  = path.join(walletPath, 'admin-key.pem');

    if (fs.existsSync(certPath) && fs.existsSync(keyPath)) {
        console.log('Admin already enrolled');
        return;
    }

    const enrollment = await ca.enroll({
        enrollmentID: 'admin',
        enrollmentSecret: 'adminpw'
    });

    fs.writeFileSync(certPath, enrollment.certificate);
    fs.writeFileSync(keyPath, enrollment.key.toBytes());

    console.log('✅ Successfully enrolled admin');
}

main().catch((error) => {
    console.error('❌ Failed to enroll admin:', error);
    process.exit(1);
});
