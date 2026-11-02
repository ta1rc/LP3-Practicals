// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/*
===============================================================================
BCT3 : Write a smart contract on a test network, for Bank account of a
       customer for following operations:
         - Deposit money
         - Withdraw Money
         - Show balance
===============================================================================

HOW TO RUN (Remix IDE - https://remix.ethereum.org)
  1. Create file Bank.sol, paste this code.
  2. Solidity Compiler tab -> select version 0.8.x -> Compile.
  3. Deploy & Run tab -> Environment:
       "Remix VM (Cancun)"         - local in-browser test chain, OR
       "Injected Provider-MetaMask" - real TEST NETWORK (e.g. Sepolia) using
                                      free test ETH from a faucet.
  4. Deploy. Then:
       - deposit()        : put e.g. 2 ether in the VALUE field, click deposit
       - showBalance()    : shows your balance in wei (1 ether = 10^18 wei)
       - withdraw(amount) : enter amount in wei, e.g. 1000000000000000000
       - depositAmount(x) / withdrawAmount(x) : "virtual" bank without real
                            ether (useful if you have no test ETH)
  5. Observe transaction hash, gas used and status in the Remix console /
     Etherscan (for Sepolia).

THEORY
------
1. BLOCKCHAIN (Unit I & II)
   A blockchain is a DISTRIBUTED, DECENTRALIZED, IMMUTABLE LEDGER of
   transactions grouped in blocks. Each block contains: block header
   (previous block hash, Merkle root of transactions, timestamp, nonce,
   difficulty) and the list of transactions. Changing any block changes its
   hash and breaks the chain -> tamper evident.
   - CRYPTOGRAPHIC HASH (SHA-256 in Bitcoin, Keccak-256 in Ethereum): one-way,
     fixed length, deterministic, avalanche effect, collision resistant.
   - MERKLE TREE: binary tree of hashes; root summarises all transactions;
     allows efficient proof of inclusion (O(log n)).
   - ASYMMETRIC CRYPTOGRAPHY / ECC: every account has a private key (signs
     transactions) and public key (verifies); Ethereum uses ECDSA on curve
     secp256k1. Address = last 20 bytes of Keccak-256(public key).
   - DIGITAL SIGNATURE: proves ownership & integrity of a transaction.
   CENTRALIZED vs DECENTRALIZED: a bank keeps a central ledger (single point
   of failure, trust needed); blockchain replicates the ledger on all nodes
   and uses CONSENSUS.
   Layers: Application, Execution (smart contracts / EVM), Semantic (rules,
   data validity), Propagation (P2P gossip), Consensus.

2. CONSENSUS (Unit III)
   Proof of Work (Bitcoin; miners solve hash puzzle), Proof of Stake
   (Ethereum since "The Merge" 2022; validators stake ETH), PBFT / Byzantine
   Generals problem, Proof of Elapsed Time, Proof of Activity, Proof of Burn.

3. ETHEREUM & SMART CONTRACTS (Unit V)
   Ethereum is a programmable, public blockchain. A SMART CONTRACT is a
   program stored at an address on the blockchain that runs exactly as coded
   when called by a transaction - no third party needed.
   - EVM (Ethereum Virtual Machine): stack-based, Turing-complete runtime;
     every node executes the bytecode deterministically.
   - Accounts: EOA (Externally Owned Account - controlled by private key, e.g.
     MetaMask) and CONTRACT accounts (controlled by code).
   - GAS: unit of computational work. Fee = gas used x gas price
     (EIP-1559: base fee + priority tip, in gwei; 1 gwei = 10^-9 ETH).
     Reading data with view functions is free when called off-chain.
   - Networks: MAINNET (real ETH), TESTNETS (Sepolia, Holesky - free test
     ETH from faucets), local (Remix VM, Ganache, Hardhat).
   - Tools: Remix IDE, Truffle suite (Truffle, Ganache, Drizzle), Hardhat,
     MetaMask wallet; Swarm (decentralized storage), Whisper (messaging).

4. SOLIDITY CONSTRUCTS USED
   - pragma          : compiler version.
   - state variables : stored permanently in contract storage.
   - mapping(address => uint) : key-value store - balance of each customer.
   - msg.sender      : address calling the function; msg.value : wei sent.
   - payable         : function / address can receive ether.
   - require(cond, "msg") : validates input; reverts the transaction (state
     rolled back, remaining gas refunded) if false.
   - event / emit    : writes logs to the blockchain, readable by front-ends.
   - modifier        : reusable pre-condition check.
   - view            : reads but does not modify state.
   - Checks-Effects-Interactions pattern: update balance BEFORE sending
     ether to prevent RE-ENTRANCY attacks.

CONCLUSION: The Bank contract lets each customer deposit, withdraw and view
balance securely; every state-changing call is a transaction that costs gas
and is permanently recorded on the test network.
===============================================================================
*/

contract Bank {
    address public owner;
    mapping(address => uint256) private balances;       // real ether (wei)
    mapping(address => uint256) private virtualBalance; // no-ether demo

    event Deposited(address indexed customer, uint256 amount);
    event Withdrawn(address indexed customer, uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    modifier positive(uint256 amount) {
        require(amount > 0, "Amount must be greater than zero");
        _;
    }

    // ---------------------------------------------------------------- Deposit
    function deposit() public payable positive(msg.value) {
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    // --------------------------------------------------------------- Withdraw
    function withdraw(uint256 amount) public positive(amount) {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;                 // effect first
        (bool ok, ) = payable(msg.sender).call{value: amount}("");
        require(ok, "Transfer failed");                 // then interaction
        emit Withdrawn(msg.sender, amount);
    }

    // ----------------------------------------------------------- Show balance
    function showBalance() public view returns (uint256) {
        return balances[msg.sender];
    }

    // total ether held by the contract (bank reserve)
    function bankReserve() public view returns (uint256) {
        return address(this).balance;
    }

    // ---------------- Same operations without sending real ether -----------
    function depositAmount(uint256 amount) public positive(amount) {
        virtualBalance[msg.sender] += amount;
        emit Deposited(msg.sender, amount);
    }

    function withdrawAmount(uint256 amount) public positive(amount) {
        require(virtualBalance[msg.sender] >= amount, "Insufficient balance");
        virtualBalance[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
    }

    function showVirtualBalance() public view returns (uint256) {
        return virtualBalance[msg.sender];
    }
}
