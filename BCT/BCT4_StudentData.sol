// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/*
===============================================================================
BCT4 : Write a program in solidity to create Student data. Use the following
       constructs:
         - Structures
         - Arrays
         - Fallback
       Deploy this as smart contract on Ethereum and Observe the transaction
       fee and Gas values.
===============================================================================

HOW TO RUN (Remix IDE - https://remix.ethereum.org)
  1. Create StudentData.sol, paste code, compile with 0.8.x.
  2. Deploy & Run -> Environment "Remix VM" (or "Injected Provider -
     MetaMask" connected to Sepolia testnet for a real network).
  3. Call addStudent(1, "Nishant", "Computer", 85) etc.
  4. Call getStudent(0), getStudentCount(), getAllStudents().
  5. FALLBACK: in "Low level interactions" at the bottom of the deployed
     contract, enter any CALLDATA such as 0x1234 and click Transact ->
     fallback() runs and emits FallbackCalled. Sending only ether with empty
     calldata triggers receive().
  6. OBSERVE GAS: expand each transaction in the Remix console and note
       - gas (gas limit), transaction cost, execution cost
       - on Sepolia/Etherscan: Gas Used, Gas Price (gwei),
         Transaction Fee = Gas Used x Gas Price.
     Typical observations (Remix VM, approximate):
       deployment        ~ 900,000 - 1,100,000 gas  (largest - stores code)
       addStudent (1st)  ~ 120,000 - 140,000 gas    (writes new storage)
       addStudent (next) ~ 100,000 gas
       view functions    -> 0 fee when called (no transaction)
     Writing new storage slots (SSTORE 0 -> non-zero = 20,000 gas each) is
     the most expensive operation; longer strings cost more gas.

THEORY
------
1. SOLIDITY
   Statically typed, contract-oriented, high-level language for writing smart
   contracts that compile to EVM bytecode. Influenced by C++, JavaScript and
   Python. A contract is similar to a class: state variables + functions +
   events + modifiers.
   Data locations: storage (persistent, on blockchain, costly), memory
   (temporary, function lifetime), calldata (read-only function arguments).
   Visibility: public, private, internal, external.
   Value types: bool, uint/int (8..256 bits), address, bytes32, enum.
   Reference types: string, arrays, struct, mapping.

2. STRUCTURE (struct)
   User-defined type grouping related variables:
       struct Student { uint rollNo; string name; string dept; uint marks; }

3. ARRAYS
   Fixed-size  T[k]  or dynamic  T[]. Members: .length, .push(), .pop().
   Here Student[] students is a dynamic array of structs in storage.
   Iterating over large arrays costs gas proportional to length - avoid
   unbounded loops in state-changing functions.

4. FALLBACK & RECEIVE FUNCTIONS
   - receive() external payable : executed when the contract receives plain
     ether with EMPTY calldata (e.g. send/transfer).
   - fallback() external [payable] : executed when a function that does not
     exist is called, or ether is sent with non-empty data and no receive.
   Neither has a name, arguments or return value. Used to accept ether, log
   unexpected calls, or implement proxy contracts.

5. GAS & TRANSACTION FEE
   Every EVM opcode has a fixed gas cost (ADD = 3, SLOAD = 2100 cold,
   SSTORE = 20,000 for new value, contract creation 32,000 + code deposit 200
   per byte). Gas limit = maximum gas user allows; unused gas is refunded.
   EIP-1559: Fee = Gas used x (Base fee + Priority fee). Base fee is burned;
   priority fee (tip) goes to the validator.
   Transaction cost = execution cost + intrinsic cost (21,000 base +
   calldata bytes).

6. ETHEREUM NETWORKS: Mainnet, testnets (Sepolia, Holesky), local dev chains
   (Remix VM, Ganache). Deployment on a public network requires a wallet
   (MetaMask) with ETH to pay gas.

CONCLUSION: The contract stores student records using a struct and a dynamic
array, handles unknown calls and plain ether with fallback/receive, and the
gas values observed show that storage writes and deployment are the costliest
operations.
===============================================================================
*/

contract StudentData {
    struct Student {
        uint256 rollNo;
        string name;
        string department;
        uint256 marks;
    }

    Student[] private students;                 // dynamic array of structs
    address public owner;

    event StudentAdded(uint256 indexed rollNo, string name);
    event FallbackCalled(address sender, uint256 value, bytes data);
    event EtherReceived(address sender, uint256 value);

    constructor() {
        owner = msg.sender;
    }

    function addStudent(uint256 _rollNo, string memory _name,
                        string memory _dept, uint256 _marks) public {
        require(_marks <= 100, "Marks must be between 0 and 100");
        for (uint256 i = 0; i < students.length; i++) {
            require(students[i].rollNo != _rollNo, "Roll number exists");
        }
        students.push(Student(_rollNo, _name, _dept, _marks));
        emit StudentAdded(_rollNo, _name);
    }

    function getStudent(uint256 index) public view
        returns (uint256, string memory, string memory, uint256)
    {
        require(index < students.length, "Invalid index");
        Student storage s = students[index];
        return (s.rollNo, s.name, s.department, s.marks);
    }

    function getStudentByRoll(uint256 _rollNo) public view
        returns (Student memory)
    {
        for (uint256 i = 0; i < students.length; i++) {
            if (students[i].rollNo == _rollNo) return students[i];
        }
        revert("Student not found");
    }

    function getAllStudents() public view returns (Student[] memory) {
        return students;
    }

    function getStudentCount() public view returns (uint256) {
        return students.length;
    }

    // called for plain ether transfers with empty calldata
    receive() external payable {
        emit EtherReceived(msg.sender, msg.value);
    }

    // called when no function matches the calldata
    fallback() external payable {
        emit FallbackCalled(msg.sender, msg.value, msg.data);
    }

    function contractBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
