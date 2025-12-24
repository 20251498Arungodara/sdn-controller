'use strict';

const { Contract } = require('fabric-contract-api');

class EpochContract extends Contract {

  async RegisterController(ctx, controllerId, pubkeyPem) {
    const key = `CONTROLLER_${controllerId}`;

    const exists = await ctx.stub.getState(key);
    if (exists && exists.length > 0) {
      throw new Error('Controller already registered');
    }

    const record = {
      controllerId,
      pubkeyPem
    };

    const value = JSON.stringify(record, Object.keys(record).sort());
    await ctx.stub.putState(key, Buffer.from(value));

    // 🔔 EVENT: controller registered
    ctx.stub.setEvent(
      'CONTROLLER_REGISTERED_EVENT',
      Buffer.from(value)
    );

    return Buffer.from('CONTROLLER_REGISTERED');
  }

  async SubmitEpoch(ctx, epochJson) {
    const epoch = JSON.parse(epochJson);

    const record = {
      epoch: Number(epoch.epoch),
      controller: String(epoch.controller),
      reward: String(epoch.reward) // deterministic
    };

    const key = `EPOCH_${record.epoch}_${record.controller}`;

    const value = JSON.stringify(record, Object.keys(record).sort());
    await ctx.stub.putState(key, Buffer.from(value));
    
    ctx.stub.setEvent(
    'EpochSubmitted',
    Buffer.from(JSON.stringify(record))
  );

    // 🔔 EVENT: epoch submitted
    //ctx.stub.setEvent(
     // 'EPOCH_SUBMITTED_EVENT',
     // Buffer.from(value)
    //);

    return Buffer.from('EPOCH_SUBMITTED');
  }

  /* =========================================================
     QUERY FUNCTIONS (READ-ONLY)
  ========================================================= */

  async QueryController(ctx, controllerId) {
    const key = `CONTROLLER_${controllerId}`;
    const data = await ctx.stub.getState(key);

    if (!data || data.length === 0) {
      throw new Error(`Controller ${controllerId} not found`);
    }

    return data;
  }

  async QueryEpoch(ctx, epoch, controllerId) {
    const key = `EPOCH_${epoch}_${controllerId}`;
    const data = await ctx.stub.getState(key);

    if (!data || data.length === 0) {
      throw new Error(`Epoch ${epoch} for controller ${controllerId} not found`);
    }

    return data;
  }
}

module.exports = EpochContract;
