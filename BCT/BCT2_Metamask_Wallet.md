# BCT2 : Create your own wallet using MetaMask for crypto transactions

This practical is a procedure, not a program. Follow the steps, take screenshots for the journal, and use the theory below for the write-up and viva.

## Procedure

### A. Install MetaMask
1. Open Chrome, Firefox, Brave or Edge and go to **https://metamask.io/download**. Only use the official site.
2. Click **Install MetaMask for <browser>** and then **Add to browser / Add extension**.
3. Pin the fox icon to the toolbar.

### B. Create the wallet
1. Click the MetaMask icon and choose **Create a new wallet**. Agree to the terms.
2. Set a strong **password**. It only unlocks MetaMask on this device.
3. **Secure your wallet.** MetaMask shows a **12-word Secret Recovery Phrase (SRP)**. Write it down on paper in the correct order and never share or photograph it.
4. Confirm the phrase by selecting the requested words. The wallet is now created.
5. **Account 1** appears with an address like `0xAbC...123`. Click it to copy. This is your public address and is safe to share for receiving funds.

### C. Switch to a test network
1. Open the network selector at the top left and turn on **Show test networks**. It is also under Settings → Advanced.
2. Select **Sepolia** test network.

### D. Get free test ETH (faucet)
1. Copy your account address.
2. Open a Sepolia faucet, for example the Google Cloud Web3 faucet, the Alchemy Sepolia faucet or the Infura faucet. Paste the address and request ETH.
3. After a short wait the balance shows, for example, `0.05 SepoliaETH`.

### E. Make a crypto transaction
1. Create a second account: **Account menu → Add account → Account 2**. You can also use a friend's address.
2. From Account 1 click **Send**, paste Account 2's address and enter an amount such as `0.001 ETH`.
3. Review the **gas fee estimate** (Low / Market / Aggressive) and the **total = amount + gas fee**.
4. Click **Confirm**. The transaction is first *Pending* and then *Confirmed*.
5. Open the **Activity** tab, click the transaction and choose **View on block explorer** (sepolia.etherscan.io). Note these values:
   - Transaction hash, status, block number and timestamp
   - From / To / Value
   - **Gas limit, Gas used, Base fee, Max priority fee, Gas price (gwei)**
   - **Transaction fee = Gas used × Gas price.** A simple ETH transfer always uses **21,000 gas**.

### F. Connect to Remix (used in BCT3 and BCT4)
In Remix, go to the Deploy tab, set Environment to **Injected Provider – MetaMask** and approve the connection. Contracts deployed this way go to Sepolia and are paid for from this wallet.

## Theory

**Cryptocurrency (Unit IV).** A cryptocurrency is a digital currency secured by cryptography and recorded on a blockchain, with no central bank. Bitcoin (2009, Satoshi Nakamoto) was the first. It uses Proof of Work, SHA-256 and the UTXO model, and has a fixed supply of 21 million. Ether (ETH) is the native currency of Ethereum. It pays for gas and uses an account-based model.

Types of cryptocurrency:
- **Coins** have their own chain (BTC, ETH).
- **Altcoins** are alternatives to Bitcoin (LTC, ADA).
- **Tokens** are built on another chain, such as ERC-20 fungible tokens (USDT, LINK).
- **NFTs** (ERC-721) are non-fungible.
- **Stablecoins** are pegged to fiat (USDC, DAI).

Uses include payments, remittance, DeFi, smart-contract fees, fundraising (ICO) and gaming.

**Crypto wallet.** A wallet does not store coins. It stores the **private keys** that control addresses on the blockchain.
- Private key: a 256-bit secret that signs transactions using ECDSA on secp256k1. Whoever holds it owns the funds.
- Public key and address: derived from the private key. An Ethereum address is the last 20 bytes of Keccak-256(public key).
- Seed phrase (BIP-39): 12 or 24 words that generate all keys deterministically (HD wallet, BIP-32/44). It is used to recover the wallet on any device.

Types of wallet:

| Basis | Types |
|---|---|
| Internet connection | **Hot wallet** (online: MetaMask, mobile apps) vs **Cold wallet** (offline: Ledger/Trezor hardware, paper wallet) |
| Key control | **Custodial** (exchange holds keys: Coinbase, Binance exchange accounts) vs **Non-custodial** (user holds keys: MetaMask) |
| Form | Browser extension, mobile, desktop, hardware, paper |

**MetaMask** is a free, non-custodial hot wallet available as a browser extension and mobile app for Ethereum and EVM chains (Polygon, BNB Chain and others). It holds ETH and tokens, signs transactions and connects dApps to the blockchain by injecting a Web3 provider (`window.ethereum`). **Coinbase** is a US exchange with a custodial exchange wallet and a separate non-custodial Coinbase Wallet. **Binance** is the largest exchange and offers a custodial exchange wallet and Trust Wallet.

**Transaction life-cycle.** The user creates a transaction (to, value, data, nonce, gas limit, fees). MetaMask signs it with the private key and broadcasts it to the P2P network. It waits in the mempool until a validator includes it in a block (Proof of Stake). Once the block is confirmed and finalised, the transaction is immutable.

**Gas (EIP-1559).** Fee = Gas used × (Base fee + Priority tip). The base fee is burned and the tip goes to the validator. Prices are quoted in gwei, where 1 gwei = 10⁻⁹ ETH. A plain transfer uses 21,000 gas, and contract calls use more.

**Security practices.**
- Never share the seed phrase or private key.
- Beware of phishing sites and fake support.
- Verify addresses before sending.
- Use a hardware wallet for large amounts.
- Revoke unused dApp approvals.

**Conclusion.** A non-custodial MetaMask wallet was created and secured with a seed phrase. Test ETH was received from a faucet and sent to another account on Sepolia, and the gas used and transaction fee were checked on Etherscan.
