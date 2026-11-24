// A tiny stack virtual machine for make.fun's tiny stack ISA v1 (ISA.md in
// the kit's generator/subject/): for each /input/<id>/{program.bin,input.bin}
// it writes /output/<id>/{stdout.bin,exit.txt}, mirroring the reference VM
// exactly: wrapping 32-bit arithmetic with an exact MUL, signed and unsigned
// divide pairs, five-bit shift counts, sign-extended PUSH8, every trap's code.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// [immediate bytes, pops, pushes] per opcode; a missing entry is no opcode (trap 5).
const META: Record<number, [number, number, number]> = {
  0x00: [0,0,0], 0x01: [1,0,1], 0x02: [4,0,1], 0x03: [0,1,0], 0x04: [0,1,2],
  0x05: [0,2,2], 0x06: [0,2,3], 0x07: [0,3,3], 0x17: [0,1,1], 0x1b: [0,1,1],
  0x30: [2,0,0], 0x31: [2,1,0], 0x32: [2,1,0], 0x33: [2,0,0], 0x34: [0,0,0],
  0x40: [0,1,1], 0x41: [0,2,0], 0x50: [0,1,0], 0x51: [0,1,0], 0x52: [0,0,1],
  0x53: [0,1,0],
};
for (let o = 0x10; o <= 0x22; o++) if (o !== 0x17 && o !== 0x1b) META[o] = [0, 2, 1];

function run(p: Uint8Array, inp: Uint8Array): { out: Uint8Array; code: number } {
  const D = new Int32Array(256); // data stack
  const C = new Int32Array(64); // call stack
  const M = new Int32Array(256); // memory
  const out = new Uint8Array(65536);
  let sp = 0, csp = 0, olen = 0, ip = 0, pc = 0, steps = 0;
  const n = p.length;
  const done = (code: number) => ({ out: out.slice(0, olen), code });
  for (;;) {
    // A step, in ISA.md's order: step limit, pc bounds, opcode, immediate fit,
    // stack underflow/overflow, then the instruction's own traps and effect.
    if (steps === 1000000) return done(10);
    if (pc < 0 || pc >= n) return done(9);
    const op = p[pc];
    const meta = META[op];
    if (meta === undefined) return done(5);
    const [imm, pops, pushes] = meta;
    if (pc + imm >= n) return done(9);
    steps++;
    if (sp < pops) return done(2);
    if (sp - pops + pushes > 256) return done(3);
    const next = pc + 1 + imm;
    pc = next;
    switch (op) {
      case 0x00: return done(0); // HALT
      case 0x01: D[sp++] = (p[next - 1] << 24) >> 24; break; // PUSH8, sign-extended
      case 0x02: D[sp++] = p[next - 4] | (p[next - 3] << 8) | (p[next - 2] << 16) | (p[next - 1] << 24); break; // PUSH32
      case 0x03: sp--; break; // POP
      case 0x04: D[sp] = D[sp - 1]; sp++; break; // DUP
      case 0x05: { const t = D[sp - 1]; D[sp - 1] = D[sp - 2]; D[sp - 2] = t; break; } // SWAP
      case 0x06: D[sp] = D[sp - 2]; sp++; break; // OVER
      case 0x07: { const a = D[sp - 3]; D[sp - 3] = D[sp - 2]; D[sp - 2] = D[sp - 1]; D[sp - 1] = a; break; } // ROT
      case 0x17: D[sp - 1] = -D[sp - 1] | 0; break; // NEG, wraps
      case 0x1b: D[sp - 1] = ~D[sp - 1]; break; // NOT
      case 0x30: case 0x31: case 0x32: { // JMP / JZ / JNZ: next + signed offset
        const rel = ((p[next - 2] | (p[next - 1] << 8)) << 16) >> 16;
        let take = op === 0x30;
        if (op !== 0x30) take = (op === 0x31) === (D[--sp] === 0);
        if (take) pc = next + rel;
        break;
      }
      case 0x33: // CALL: the immediate is the unsigned target address
        if (csp === 64) return done(7);
        C[csp++] = next;
        pc = p[next - 2] | (p[next - 1] << 8);
        break;
      case 0x34: // RET
        if (csp === 0) return done(8);
        pc = C[--csp];
        break;
      case 0x40: { // LOAD
        const a = D[sp - 1];
        if (a < 0 || a > 255) return done(6);
        D[sp - 1] = M[a];
        break;
      }
      case 0x41: { // STORE: ( v addr -- )
        const a = D[sp - 1];
        if (a < 0 || a > 255) return done(6);
        M[a] = D[sp - 2];
        sp -= 2;
        break;
      }
      case 0x50: { // PRINT: decimal; past the output limit it writes nothing
        const s = String(D[sp - 1]);
        if (olen + s.length > 65536) return done(11);
        for (let i = 0; i < s.length; i++) out[olen++] = s.charCodeAt(i);
        sp--;
        break;
      }
      case 0x51: // PUTC: one byte, the value's low eight bits
        if (olen === 65536) return done(11);
        out[olen++] = D[--sp] & 255;
        break;
      case 0x52: D[sp++] = ip < inp.length ? inp[ip++] : -1; break; // IN, -1 once used up
      case 0x53: return done(D[--sp] & 255); // EXIT: the code is the value's low byte
      default: { // the binary operators: ( a b -- r ), b on top
        const b = D[sp - 1], a = D[sp - 2];
        let r: number;
        switch (op) {
          case 0x10: r = (a + b) | 0; break; // ADD
          case 0x11: r = (a - b) | 0; break; // SUB
          case 0x12: r = Math.imul(a, b); break; // MUL: the exact low 32 bits
          case 0x13: if (b === 0) return done(4); r = a === -2147483648 && b === -1 ? a : (a / b) | 0; break; // DIV
          case 0x14: if (b === 0) return done(4); r = b === -1 ? 0 : (a % b) | 0; break; // MOD
          case 0x15: if (b === 0) return done(4); r = Math.floor((a >>> 0) / (b >>> 0)) | 0; break; // DIVU
          case 0x16: if (b === 0) return done(4); r = ((a >>> 0) % (b >>> 0)) | 0; break; // MODU
          case 0x18: r = a & b; break;
          case 0x19: r = a | b; break;
          case 0x1a: r = a ^ b; break;
          case 0x1c: r = a << (b & 31); break; // counts use five bits
          case 0x1d: r = a >> (b & 31); break;
          case 0x1e: r = (a >>> (b & 31)) | 0; break;
          case 0x20: r = a === b ? 1 : 0; break;
          case 0x21: r = a < b ? 1 : 0; break;
          default: r = (a >>> 0) < (b >>> 0) ? 1 : 0; break; // LTU
        }
        D[sp - 2] = r;
        sp--;
      }
    }
  }
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(I).sort()) {
  const r = run(readFileSync(join(I, id, 'program.bin')), readFileSync(join(I, id, 'input.bin')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'stdout.bin'), r.out);
  writeFileSync(join(O, id, 'exit.txt'), `${r.code}\n`);
}
