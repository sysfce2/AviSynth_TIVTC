/*
**   Helper methods for TIVTC and TDeint
**
**   Copyright (C)2026 pinterf
**
**   This program is free software; you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation; either version 2 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program; if not, write to the Free Software
**   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/


// SIMD-accelerated AnalyzeDiffMask for planar, DIST=1.
//
//   SSE4.1 path : 16 pixels / iteration
//   AVX2   path : 32 pixels / iteration
//
// uint16 (10/12/14/16-bit) variants: not now
// YUY2 (DIST=2/4) is unchanged from the original scalar code.
//
// ─────────────────────────────────────────────────────────────────────────────
// OVERVIEW: TWO-PHASE APPROACH
// ─────────────────────────────────────────────────────────────────────────────
// The original AnalyzeOnePixel() is called once per pixel and performs several
// threshold comparisons against the same five source rows on every call.
// Converting those comparisons to SIMD directly is awkward because the inputs
// are reused across multiple logical steps.
//
// Instead, work is split into two phases:
//
//   Phase 1 — BuildBoolRow_u8 (once per source row, amortized):
//     Each source row is converted into two boolean rows:
//       gt3[x]  = 0xFF if src[x] >  3, else 0x00   (Const3  = 3)
//       gt19[x] = 0xFF if src[x] > 19, else 0x00   (Const19 = 19)
//     These 0x00/0xFF values act as branchless SIMD booleans: they can be
//     ORed, ANDed, and summed directly without any further comparisons.
//     A 5-slot rotating ring buffer (BoolRowRing) holds the boolean rows for
//     dppp/dpp/dp/dpn/dpnn.  On each row advance only the new dpnn slot is
//     rebuilt; the other four are kept by rotating the pointer array.
//     Padding (BROW_PAD bytes of zeros on each side) is written once in
//     alloc() and never touched again, eliminating 10 memset calls per row.
//
//   Phase 2 — ProcessRow_SSE41 / ProcessRow_AVX2 (hot loop):
//     Reads the precomputed boolean rows and computes per-pixel increments
//     (+0, +1, +2, +4, +5) as branchless SIMD masks, then writes them back
//     to dstp with a single PADDUSB.
//
// ─────────────────────────────────────────────────────────────────────────────
// CONCEPTUAL SIMD WORKFLOW (SSE4.1, 16 pixels at a time)
// ─────────────────────────────────────────────────────────────────────────────
//
//  Step  Operation                  SIMD implementation
//  ────  ─────────────────────────  ──────────────────────────────────────────
//   A    Center pixel > 3?          Load gt3_dp[x]; _mm_testz -> skip block
//   B    Any 3x3 neighbor > 3?      OR 8 loads from gt3_dpp/dp/dpn; testz
//   C    Base increment +1          _mm_and_si128(active, one)
//   D    Center pixel > 19?         Load gt19_dp[x]; testz -> flush +1 only
//   E    Count 3x3 ring > 19        AND each of 8 gt19 loads with 0x01,
//                                   sum with _mm_add_epi8 -> edi (0..8)
//        edi > 2?                   _mm_cmpgt_epi8(edi, two); testz -> flush
//        Quick path: upper && lower _mm_and of upper_quick + lower_quick
//                    -> +2, skip F  masks -> add one more to inc
//   F    Wide scan +/-4 in dpp/dpn  horzOR9_epu8: 8 x PALIGNR + OR gives
//        (and dppp/dpnn for upper2/ per-lane OR of 9 consecutive boolean
//        lower2)                    bytes without any scalar loop
//        Logic blending             PCMPEQ/PCMPGT/PANDN/POR on upper_w,
//                                   lower_w, upper2_w, lower2_w, edi_gt4
//                                   -> add2 mask and add4 mask
//        Apply deltas               _mm_adds_epu8(inc, and(add2, one))
//                                   _mm_adds_epu8(inc, and(add4, three))
//   G    Write back                 _mm_adds_epu8(dstp[x], inc) -> storeu
//
// The AVX2 path is structurally identical, processing 32 pixels per iteration
// using ymm (_mm256_*) equivalents of every operation above.
//
// ─────────────────────────────────────────────────────────────────────────────
// horzOR9_epu8 — ALWAYS XMM, EVEN IN AVX2 CONTEXT
// ─────────────────────────────────────────────────────────────────────────────
// The original scalar loop  for (s = x-4; s < x+5; s++) if (row[s] > 19) ...
// is replaced by ORing 9 byte-shifted copies of the boolean row:
//
//   lo = brow[x-4 .. x+11]    hi = brow[x+4 .. x+19]
//   acc = lo | alignr(hi,lo,1) | alignr(hi,lo,2) | ... | alignr(hi,lo,8)
//
// alignr(hi, lo, s) places byte (x+k-4+s) in lane k for all 16 lanes
// simultaneously, so acc[k] = OR of brow[x+k-4 .. x+k+4] — exactly the
// 9-byte window needed for DIST=1.
//
// IMPORTANT: _mm256_alignr_epi8 (vpalignr ymm) operates on TWO INDEPENDENT
// 128-bit lanes and cannot slide bytes across the 128-bit boundary.  Using it
// naively for a byte-sliding window would silently corrupt lanes 12–19 of a
// 32-byte vector.  The fix is simple: horzOR9_epu8 always works on xmm, and
// in the AVX2 path horzOR9_x32 calls it twice (at x and x+16) and reassembles
// the result with _mm256_set_m128i.  The two extra xmm calls cost ~2 cycles
// and are far preferable to any cross-lane workaround.
//
// ─────────────────────────────────────────────────────────────────────────────
// CORRECTNESS NOTES
// ─────────────────────────────────────────────────────────────────────────────
//  - The "quick path" (count≠esi && upper≠0 -> +2; return) in the original
//    uses the 3x3-only upper/lower flags, NOT the wide scan values.
//    Reproduced exactly: upper_quick = OR of dpp[x-1..x+1] > 19 only.
//
//  - Step F resets upper and lower from the +/-4 wide scan, overwriting the
//    quick-path values.  The original does the same by declaring new local
//    variables.  Pixels that took the quick path are excluded from step F
//    via the needScan mask, so there is no interaction.
//
//  - upper2 (dppp scan) is forced to zero when y == 2 (dppp row is outside
//    the valid frame); lower2 (dpnn scan) is forced to zero when
//    y == Height-4.  Both are represented as conditional ternary
//    assignments before the scan logic, producing z/z256 for those rows.
//
//  - Edge pixels x < 5 and x >= Width-5 always go through the scalar
//    AnalyzeOnePixel_D1 replica.  This avoids out-of-bounds loads in
//    horzOR9_epu8 (which reaches x-4 and x+4+15) without any masking cost
//    in the hot loop.  For typical HD/4K widths the edge region is <1% of
//    total pixels.
//
//  - _mm_adds_epu8 (saturating add) is used throughout for dstp writes.
//    In normal operation dstp values stay well below 255, but saturation
//    prevents corruption on pathological inputs at zero extra cost.
//
// ─────────────────────────────────────────────────────────────────────────────
// BACKGROUND: tbuffer and the two thresholds
//
// tbuffer is a pre-computed difference buffer filled by the caller before
// AnalyzeDiffMask is invoked.  Each element tbuffer[row][x] holds the
// absolute difference between the two fields at that sample position —
// effectively a per-pixel measure of how much the two interlaced fields
// disagree at (row, x).  Values are in the same bit-depth as the source
// (0..255 for 8-bit, scaled for 10/12/14/16-bit).
//
// The algorithm classifies each difference value against two fixed thresholds:
//
//   Threshold 3  (C3  = 3 << (bpp-8)):
//     The noise floor.  Differences at or below this level are indistinguishable
//     from sensor noise, compression ringing, or sub-LSB chroma bleed.
//     A pixel whose diff <= 3 is treated as "fields agree here" and skipped
//     entirely (steps A and B).  For 8-bit material this means any diff of
//     0..3 is ignored.
//
//   Threshold 19  (C19 = 19 << (bpp-8)):
//     The combing signal threshold.  A diff > 19 indicates enough contrast
//     between the two fields at this sample to be structurally significant —
//     a likely comb tooth rather than soft motion or texture.  The value 19
//     is hardcoded from the original TIVTC implementation (tritical); it is
//     the empirically tuned boundary for standard-dynamic-range interlaced
//     content.  Pixels above this threshold drive the neighbourhood counting
//     (edi), directional flags (upper/lower/upper2/lower2), and the +2/+4
//     increments written to dstp.
//
//   Both thresholds scale linearly with bit depth via the (bpp-8) shift so
//   that the same perceptual boundary is preserved regardless of sample depth.
//   In this file the 8-bit SIMD path hardcodes the literal values 3 and 19
//   because BuildBoolRow_u8 only handles uint8; AnalyzeOnePixel_D1 uses the
//   scaled constexpr C3/C19 for the templated uint16 paths.
//
// The dstp output accumulates a "combing score" per pixel across multiple
// calls.  The increments written here (+1, +3, +5) are intentionally
// asymmetric: +1 for any active pixel, +3 when the neighbourhood confirms
// combing in at least one direction, +5 when it is confirmed in both vertical
// directions with sufficient density (count > 4).
// ─────────────────────────────────────────────────────────────────────────────


#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>   // SSE4.1, AVX2
#include "internal.h" // _aligned_malloc, _aligned_free

// This header is included across multiple translation units (TUs) with varying 
// instruction set architectures (e.g., AVX2 vs. SSE2).
//
// To support specialized implementations, the caller defines macros (e.g., 
// INCLUDE_PROCESSROW_AVX2) before inclusion. We use an anonymous namespace 
// to grant these functions internal linkage. 
//
// This prevents the linker from performing COMDAT folding, which would otherwise
// risk merging different SIMD versions of BuildBoolRow_u16 into a single 
// implementation, potentially leading to illegal instruction crashes on 
// older hardware.

namespace {

  // ─────────────────────────────────────────────────────────────────────────────
  // Padding on each side of boolean rows.
  // Must be >= 4 (wide-scan reach) + 32 (max SIMD load width).
  // ─────────────────────────────────────────────────────────────────────────────
  constexpr int BROW_PAD = 48;

  // ─────────────────────────────────────────────────────────────────────────────
  // BuildBoolRow_u8
  // Converts one uint8 source row into two 0x00/0xFF boolean rows:
  //   gt3[x]  = 0xFF if src[x] >  3, else 0x00
  //   gt19[x] = 0xFF if src[x] > 19, else 0x00
  //
  // Padding regions (gt3[-BROW_PAD..-1] and gt3[width..width+BROW_PAD-1]) are
  // NOT zeroed here — they are zeroed once for the entire allocation in
  // BoolRowRing::alloc() and remain zero permanently because BuildBoolRow_u8
  // only ever writes to [0..width-1].  The ring-buffer rotation reuses the same
  // physical memory slots, so the invariant is maintained across all iterations.
  // ─────────────────────────────────────────────────────────────────────────────
  void BuildBoolRow_u8(const uint8_t* src, int width,
    uint8_t* gt3,   // gt3[-BROW_PAD] .. gt3[width+BROW_PAD-1] are valid
    uint8_t* gt19, int C3, int C19)
  {
    const __m128i c3 = _mm_set1_epi8((char)C3);
    const __m128i c19 = _mm_set1_epi8((char)C19);
    const __m128i z = _mm_setzero_si128();
    const __m128i ff = _mm_set1_epi8(-1);

    int x = 0;
    for (; x + 16 <= width; x += 16) {
      __m128i v = _mm_loadu_si128((const __m128i*)(src + x));
      // v > c  <=>  subs_epu8(v, c) != 0   (correct for all unsigned values)
      __m128i g3 = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(v, c3), z), ff);
      __m128i g19 = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(v, c19), z), ff);
      _mm_storeu_si128((__m128i*)(gt3 + x), g3);
      _mm_storeu_si128((__m128i*)(gt19 + x), g19);
    }
    for (; x < width; x++) {
      gt3[x] = (src[x] > 3) ? 0xFF : 0x00;
      gt19[x] = (src[x] > 19) ? 0xFF : 0x00;
    }
    // No memset of padding here — see comment above.
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // BuildBoolRow_u16
  // Converts one uint16 source row into two 0x00/0xFF boolean rows.
  // C3  = 3  << (bits_per_pixel - 8)
  // C19 = 19 << (bits_per_pixel - 8)
  //
  // Uses subs_epu16 + cmpeq (same trick as u8) so it is correct for all
  // unsigned values including full 16-bit depth where signed cmpgt would fail.
  // Processes 16 uint16 pixels per iteration (one 256-bit load), packs the
  // resulting 16 x epi16 booleans down to 16 x epi8 for the boolean row.
  // The output format (0x00/0xFF bytes) is identical to BuildBoolRow_u8, so
  // ProcessRow_SSE41 / ProcessRow_AVX2 require zero changes.
  // ─────────────────────────────────────────────────────────────────────────────
  void BuildBoolRow_u16(const uint16_t* src, int width,
    uint8_t* gt3, uint8_t* gt19,
    int C3, int C19)
  {
    int x = 0;
#ifdef INCLUDE_PROCESSROW_AVX2
    const __m256i c3 = _mm256_set1_epi16((short)C3);
    const __m256i c19 = _mm256_set1_epi16((short)C19);
    const __m256i z256 = _mm256_setzero_si256();
    const __m256i all1 = _mm256_set1_epi16(-1);
    for (; x + 16 <= width; x += 16) {
      __m256i v = _mm256_loadu_si256((const __m256i*)(src + x));
      // v > c  <=>  subs_epu16(v, c) != 0  (correct for all unsigned values)
      __m256i sat3 = _mm256_subs_epu16(v, c3);
      __m256i sat19 = _mm256_subs_epu16(v, c19);
      // cmpeq gives 0xFFFF where NOT > threshold; andnot flips to get
      // 0xFFFF where IS > threshold, 0x0000 elsewhere.
      __m256i b3 = _mm256_andnot_si256(_mm256_cmpeq_epi16(sat3, z256), all1);
      __m256i b19 = _mm256_andnot_si256(_mm256_cmpeq_epi16(sat19, z256), all1);

      // Shift high byte to low byte (0xFFFF -> 0x00FF) so packus works
      __m128i lo3 = _mm_srli_epi16(_mm256_castsi256_si128(b3), 8);
      __m128i hi3 = _mm_srli_epi16(_mm256_extracti128_si256(b3, 1), 8);
      // Pack 16 x epi16 (0x0000 or 0x00FF) -> 16 x epi8 (0x00 or 0xFF).
      __m128i lo19 = _mm_srli_epi16(_mm256_castsi256_si128(b19), 8);
      __m128i hi19 = _mm_srli_epi16(_mm256_extracti128_si256(b19, 1), 8);

      _mm_storeu_si128((__m128i*)(gt3 + x), _mm_packus_epi16(lo3, hi3));
      _mm_storeu_si128((__m128i*)(gt19 + x), _mm_packus_epi16(lo19, hi19));
    }
#endif
    // SSE2 tail: 8 pixels at a time
    const __m128i c3x = _mm_set1_epi16((short)C3);
    const __m128i c19x = _mm_set1_epi16((short)C19);
    const __m128i zx = _mm_setzero_si128();
    const __m128i all1x = _mm_set1_epi16(-1);
    for (; x + 8 <= width; x += 8) {
      __m128i v = _mm_loadu_si128((const __m128i*)(src + x));
      __m128i b3 = _mm_andnot_si128(_mm_cmpeq_epi16(_mm_subs_epu16(v, c3x), zx), all1x);
      __m128i b19 = _mm_andnot_si128(_mm_cmpeq_epi16(_mm_subs_epu16(v, c19x), zx), all1x);
      // Shift high byte to low byte (0xFFFF -> 0x00FF) so packus works
      _mm_storel_epi64((__m128i*)(gt3 + x), _mm_packus_epi16(_mm_srli_epi16(b3, 8), zx));
      _mm_storel_epi64((__m128i*)(gt19 + x), _mm_packus_epi16(_mm_srli_epi16(b19, 8), zx));
    }
    for (; x < width; x++) {
      gt3[x] = (src[x] > (uint16_t)C3) ? 0xFF : 0x00;
      gt19[x] = (src[x] > (uint16_t)C19) ? 0xFF : 0x00;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // horzOR9_epu8  — always xmm, even in AVX2 context
  //
  // For each lane k in [0..15]:
  //   result[k] = brow[x+k-4] | … | brow[x+k+4]    (9 bytes, DIST=1 reach)
  //
  // Uses 8 x alignr.  brow must have BROW_PAD >= 20 of valid (zeroed) memory
  // on each side so the two unaligned loads are always safe.
  // ─────────────────────────────────────────────────────────────────────────────
#if !defined(__AVX2__)
#if defined(GCC) || defined(CLANG)
  __attribute__((__target__("sse4.1")))
#endif
#endif
  inline __m128i horzOR9_epu8(const uint8_t* brow, int x)
  {
    // To extract a sliding window across 16 lanes, lo and hi MUST
    // represent a contiguous 32-byte block of memory.
    // lo = [x-4 ... x+11]
    // hi = [x+12 ... x+27]
    __m128i lo = _mm_loadu_si128((const __m128i*)(brow + x - 4));
    __m128i hi = _mm_loadu_si128((const __m128i*)(brow + x + 12));

    __m128i acc = lo; // s=0
    acc = _mm_or_si128(acc, _mm_alignr_epi8(hi, lo, 1));
    acc = _mm_or_si128(acc, _mm_alignr_epi8(hi, lo, 2));
    acc = _mm_or_si128(acc, _mm_alignr_epi8(hi, lo, 3));
    acc = _mm_or_si128(acc, _mm_alignr_epi8(hi, lo, 4));
    acc = _mm_or_si128(acc, _mm_alignr_epi8(hi, lo, 5));
    acc = _mm_or_si128(acc, _mm_alignr_epi8(hi, lo, 6));
    acc = _mm_or_si128(acc, _mm_alignr_epi8(hi, lo, 7));
    acc = _mm_or_si128(acc, _mm_alignr_epi8(hi, lo, 8));
    return acc;
  }

  // AVX2 helper: 32-lane horzOR9 by calling the xmm version twice.
#ifdef INCLUDE_PROCESSROW_AVX2
  // Three contiguous 128-bit blocks cover the full 32+8 byte window needed:
  //   block0: [x-4  .. x+11]  — lo for lower half
  //   block1: [x+12 .. x+27]  — hi for lower half AND lo for upper half (shared)
  //   block2: [x+28 .. x+43]  — hi for upper half
  // Saves one load vs calling horzOR9_epu8 twice independently.
  inline __m256i horzOR9_x32(const uint8_t* brow, int x)
  {
    // Load 3 contiguous 128-bit chunks
    // Total footprint: (x-4) to (x+28+15) = x-4 to x+43
    __m128i b0 = _mm_loadu_si128((const __m128i*)(brow + x - 4));
    __m128i b1 = _mm_loadu_si128((const __m128i*)(brow + x + 12));
    __m128i b2 = _mm_loadu_si128((const __m128i*)(brow + x + 28));

    // Lower 16 lanes: Sliding window across [b0 : b1]
    __m128i r_lo = b0;
    r_lo = _mm_or_si128(r_lo, _mm_alignr_epi8(b1, b0, 1));
    r_lo = _mm_or_si128(r_lo, _mm_alignr_epi8(b1, b0, 2));
    r_lo = _mm_or_si128(r_lo, _mm_alignr_epi8(b1, b0, 3));
    r_lo = _mm_or_si128(r_lo, _mm_alignr_epi8(b1, b0, 4));
    r_lo = _mm_or_si128(r_lo, _mm_alignr_epi8(b1, b0, 5));
    r_lo = _mm_or_si128(r_lo, _mm_alignr_epi8(b1, b0, 6));
    r_lo = _mm_or_si128(r_lo, _mm_alignr_epi8(b1, b0, 7));
    r_lo = _mm_or_si128(r_lo, _mm_alignr_epi8(b1, b0, 8));

    // Upper 16 lanes: Sliding window across [b1 : b2]
    // Starts at x+16-4 = x+12 (which is exactly b1)
    __m128i r_hi = b1;
    r_hi = _mm_or_si128(r_hi, _mm_alignr_epi8(b2, b1, 1));
    r_hi = _mm_or_si128(r_hi, _mm_alignr_epi8(b2, b1, 2));
    r_hi = _mm_or_si128(r_hi, _mm_alignr_epi8(b2, b1, 3));
    r_hi = _mm_or_si128(r_hi, _mm_alignr_epi8(b2, b1, 4));
    r_hi = _mm_or_si128(r_hi, _mm_alignr_epi8(b2, b1, 5));
    r_hi = _mm_or_si128(r_hi, _mm_alignr_epi8(b2, b1, 6));
    r_hi = _mm_or_si128(r_hi, _mm_alignr_epi8(b2, b1, 7));
    r_hi = _mm_or_si128(r_hi, _mm_alignr_epi8(b2, b1, 8));

    return _mm256_set_m128i(r_hi, r_lo);
  }
#endif

  // ─────────────────────────────────────────────────────────────────────────────
  // Scalar fallback — exact replica of AnalyzeOnePixel<pixel_t, bpp, DIST=1>
  // ─────────────────────────────────────────────────────────────────────────────
  template<typename pixel_t>
  inline void AnalyzeOnePixel_D1(
    uint8_t* dstp,
    const pixel_t* dppp, const pixel_t* dpp,
    const pixel_t* dp,
    const pixel_t* dpn, const pixel_t* dpnn,
    int x, int y, int Width, int Height, int C3, int C19)
  {

    if (dp[x] <= C3) return;
    if (dp[x - 1] <= C3 && dp[x + 1] <= C3 &&
      dpp[x - 1] <= C3 && dpp[x] <= C3 && dpp[x + 1] <= C3 &&
      dpn[x - 1] <= C3 && dpn[x] <= C3 && dpn[x + 1] <= C3)
      return;

    dstp[x]++;
    if (dp[x] <= C19) return;

    int edi = 0;
    if (dpp[x - 1] > C19) edi++;
    if (dpp[x] > C19) edi++;
    if (dpp[x + 1] > C19) edi++;
    int upper = (edi != 0) ? 1 : 0;
    if (dp[x - 1] > C19) edi++;
    if (dp[x + 1] > C19) edi++;
    int esi = edi;
    if (dpn[x - 1] > C19) edi++;
    if (dpn[x] > C19) edi++;
    if (dpn[x + 1] > C19) edi++;
    if (edi <= 2) return;

    int count = edi;
    // lower carries into the wide scan: if count!=esi, at least one dpn pixel
    // in the 3x3 was > C19, so lower is already 1 before the wide scan.
    // upper is NOT reset here — if count==esi (no dpn contribution) upper
    // retains its value from the 3x3 ring and the wide scan only ORs more in.
    // If count!=esi and upper!=0 we return early; if we fall through, upper
    // was 0, so the wide scan starts correctly with upper==0 either way.
    int lower = 0;
    if (count != esi) {
      lower = 1;
      if (upper != 0) { dstp[x] += 2; return; }
    }

    // Wide scan: upper and lower are accumulated (never cleared) from here.
    int startx = (x < 4) ? 0 : x - 4;
    int stopx = (x + 5 > Width) ? Width : x + 5;
    int upper2 = 0, lower2 = 0;

    if (y != 2) {
      for (int s = startx; s < stopx; s++)
        if (dppp[s] > C19) { upper2 = 1; break; }
    }
    for (int s = startx; s < stopx; s++) {
      if (dpp[s] > C19) upper = 1;
      if (dpn[s] > C19) lower = 1;
      if (upper && lower) break;
    }
    if (y != Height - 4) {
      for (int s = startx; s < stopx; s++)
        if (dpnn[s] > C19) { lower2 = 1; break; }
    }

    if (upper == 0) {
      if (lower == 0 || lower2 == 0) { if (count > 4) dstp[x] += 4; }
      else { dstp[x] += 2; }
    }
    else {
      if (lower != 0 || upper2 != 0) { dstp[x] += 2; }
      else { if (count > 4) dstp[x] += 4; }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // ProcessRow_SSE41 — 16 pixels / iteration
  // ─────────────────────────────────────────────────────────────────────────────

#if !defined(__AVX2__)
#if defined(GCC) || defined(CLANG)
  __attribute__((__target__("sse4.1")))
#endif
#endif
    void ProcessRow_SSE41(
    const uint8_t* gt3_dppp, const uint8_t* gt19_dppp,
    const uint8_t* gt3_dpp, const uint8_t* gt19_dpp,
    const uint8_t* gt3_dp, const uint8_t* gt19_dp,
    const uint8_t* gt3_dpn, const uint8_t* gt19_dpn,
    const uint8_t* gt3_dpnn, const uint8_t* gt19_dpnn,
    uint8_t* dstp,
    int xstart, int xstop,
    int y, int Width, int Height)
  {

    const __m128i z = _mm_setzero_si128();
    const __m128i ff = _mm_set1_epi8(-1);
    const __m128i one = _mm_set1_epi8(1);
    const __m128i two = _mm_set1_epi8(2);
    const __m128i c4 = _mm_set1_epi8(4);

    for (int x = xstart; x + 16 <= xstop; x += 16)
    {

      // A: dp[x] > 3
      __m128i maskA = _mm_loadu_si128((const __m128i*)(gt3_dp + x));
      if (_mm_testz_si128(maskA, maskA)) continue;
      // B: at least one 3x3 neighbor > 3
      __m128i anyNbr3 = _mm_or_si128(
        _mm_or_si128(
          _mm_or_si128(_mm_loadu_si128((const __m128i*)(gt3_dpp + x - 1)),
            _mm_loadu_si128((const __m128i*)(gt3_dpp + x))),
          _mm_or_si128(_mm_loadu_si128((const __m128i*)(gt3_dpp + x + 1)),
            _mm_loadu_si128((const __m128i*)(gt3_dp + x - 1)))),
        _mm_or_si128(
          _mm_or_si128(_mm_loadu_si128((const __m128i*)(gt3_dp + x + 1)),
            _mm_loadu_si128((const __m128i*)(gt3_dpn + x - 1))),
          _mm_or_si128(_mm_loadu_si128((const __m128i*)(gt3_dpn + x)),
            _mm_loadu_si128((const __m128i*)(gt3_dpn + x + 1)))));
      __m128i active = _mm_and_si128(maskA, anyNbr3);
      if (_mm_testz_si128(active, active)) continue;
      // C: +1 for active pixels
      __m128i inc = _mm_and_si128(active, one);
      // D: dp[x] > 19?
      __m128i center19 = _mm_loadu_si128((const __m128i*)(gt19_dp + x));
      __m128i activeD = _mm_and_si128(active, center19);
      if (_mm_testz_si128(activeD, activeD)) {
        __m128i cur = _mm_loadu_si128((const __m128i*)(dstp + x));
        _mm_storeu_si128((__m128i*)(dstp + x), _mm_adds_epu8(cur, inc));
        continue;
      }
      // E: count 3x3 ring pixels > 19
      __m128i dpp_m1 = _mm_loadu_si128((const __m128i*)(gt19_dpp + x - 1));
      __m128i dpp_0 = _mm_loadu_si128((const __m128i*)(gt19_dpp + x));
      __m128i dpp_p1 = _mm_loadu_si128((const __m128i*)(gt19_dpp + x + 1));
      __m128i dpn_m1 = _mm_loadu_si128((const __m128i*)(gt19_dpn + x - 1));
      __m128i dpn_0 = _mm_loadu_si128((const __m128i*)(gt19_dpn + x));
      __m128i dpn_p1 = _mm_loadu_si128((const __m128i*)(gt19_dpn + x + 1));
      __m128i dp_m1 = _mm_loadu_si128((const __m128i*)(gt19_dp + x - 1));
      __m128i dp_p1 = _mm_loadu_si128((const __m128i*)(gt19_dp + x + 1));
      // upper_quick / lower_quick: any of dpp/dpn 3x3 row > 19
      __m128i upper_quick = _mm_or_si128(_mm_or_si128(dpp_m1, dpp_0), dpp_p1);
      __m128i lower_quick = _mm_or_si128(_mm_or_si128(dpn_m1, dpn_0), dpn_p1);
      // edi count (0..8): convert 0xFF->1 via &1, then sum
      auto b1 = [](__m128i m) { return _mm_and_si128(m, _mm_set1_epi8(1)); };
      __m128i edi = _mm_add_epi8(_mm_add_epi8(b1(dpp_m1), b1(dpp_0)), b1(dpp_p1));
      __m128i esi_cnt = _mm_add_epi8(edi, _mm_add_epi8(b1(dp_m1), b1(dp_p1)));
      edi = _mm_add_epi8(esi_cnt,
        _mm_add_epi8(_mm_add_epi8(b1(dpn_m1), b1(dpn_0)), b1(dpn_p1)));
      __m128i edi_gt2 = _mm_cmpgt_epi8(edi, two);   // signed ok: values 0..8
      __m128i activeE = _mm_and_si128(activeD, edi_gt2);

      if (_mm_testz_si128(activeE, activeE)) {
        __m128i cur = _mm_loadu_si128((const __m128i*)(dstp + x));
        _mm_storeu_si128((__m128i*)(dstp + x), _mm_adds_epu8(cur, inc));
        continue;
      }

      // Quick path: lower_quick && upper_quick -> scalar does dstp[x]++ then +=2, total +3
      // inc already holds +1 (base); add 2 more to reach +3.
      __m128i quickPath = _mm_and_si128(activeE, _mm_and_si128(lower_quick, upper_quick));
      inc = _mm_adds_epu8(inc, _mm_and_si128(quickPath, two));

      __m128i needScan = _mm_andnot_si128(quickPath, activeE);

      if (!_mm_testz_si128(needScan, needScan))
      {
        // F: wide +/-4 scan
        __m128i upper_w = horzOR9_epu8(gt19_dpp, x);
        __m128i lower_w = horzOR9_epu8(gt19_dpn, x);
        __m128i upper2_w = (y != 2) ? horzOR9_epu8(gt19_dppp, x) : z;
        __m128i lower2_w = (y != Height - 4) ? horzOR9_epu8(gt19_dpnn, x) : z;
        __m128i edi_gt4 = _mm_cmpgt_epi8(edi, c4);
        __m128i upper_zero = _mm_cmpeq_epi8(upper_w, z);
        __m128i upper_nz = _mm_andnot_si128(upper_zero, ff);
        __m128i lower_zero = _mm_cmpeq_epi8(lower_w, z);
        __m128i lower2_zero = _mm_cmpeq_epi8(lower2_w, z);
        __m128i lower_nz = _mm_andnot_si128(lower_zero, ff);
        __m128i upper2_nz = _mm_andnot_si128(_mm_cmpeq_epi8(upper2_w, z), ff);
        // upper==0, (lower==0 || lower2==0), count>4  -> +4
        __m128i lo_or_lo2_zero = _mm_or_si128(lower_zero, lower2_zero);
        __m128i add4_u0 = _mm_and_si128(_mm_and_si128(upper_zero, lo_or_lo2_zero), edi_gt4);
        // upper==0, lower!=0 && lower2!=0             -> +2
        __m128i add2_u0 = _mm_and_si128(upper_zero, _mm_andnot_si128(lo_or_lo2_zero, ff));
        // upper!=0, (lower!=0 || upper2!=0)           -> +2
        __m128i lo_or_u2 = _mm_or_si128(lower_nz, upper2_nz);
        __m128i add2_u1 = _mm_and_si128(upper_nz, lo_or_u2);
        // upper!=0, lower==0 && upper2==0, count>4   -> +4
        __m128i add4_u1 = _mm_and_si128(_mm_and_si128(upper_nz,
          _mm_andnot_si128(lo_or_u2, ff)), edi_gt4);
        __m128i add2 = _mm_and_si128(_mm_or_si128(add2_u0, add2_u1), needScan);
        __m128i add4 = _mm_and_si128(_mm_or_si128(add4_u0, add4_u1), needScan);
        // Wide scan outcomes — inc already holds +1 (base); add on top:
        // +3 total (scalar's dstp[x]++ + +=2): add 2 more
        inc = _mm_adds_epu8(inc, _mm_and_si128(add2, two));
        // +5 total (scalar's dstp[x]++ + +=4): add 4 more
        inc = _mm_adds_epu8(inc, _mm_and_si128(add4, _mm_set1_epi8(4)));
      }
      __m128i cur = _mm_loadu_si128((const __m128i*)(dstp + x));
      _mm_storeu_si128((__m128i*)(dstp + x), _mm_adds_epu8(cur, inc));
    }
  }


  // ─────────────────────────────────────────────────────────────────────────────
  // ProcessRow_AVX2 — 32 pixels / iteration
  // horzOR9 stays in xmm; called twice and merged into ymm via set_m128i.
  // ─────────────────────────────────────────────────────────────────────────────
#ifdef INCLUDE_PROCESSROW_AVX2
  void ProcessRow_AVX2(
    const uint8_t* gt3_dppp, const uint8_t* gt19_dppp,
    const uint8_t* gt3_dpp, const uint8_t* gt19_dpp,
    const uint8_t* gt3_dp, const uint8_t* gt19_dp,
    const uint8_t* gt3_dpn, const uint8_t* gt19_dpn,
    const uint8_t* gt3_dpnn, const uint8_t* gt19_dpnn,
    uint8_t* dstp,
    int xstart, int xstop,
    int y, int Width, int Height)
  {
    const __m256i z256 = _mm256_setzero_si256();
    const __m256i ff256 = _mm256_set1_epi8(-1);
    const __m256i one = _mm256_set1_epi8(1);
    const __m256i two = _mm256_set1_epi8(2);
    const __m256i c4 = _mm256_set1_epi8(4);

    auto ld = [](const uint8_t* p, int x) {
      return _mm256_loadu_si256((const __m256i*)(p + x));
      };
    auto b1 = [](__m256i m) {
      return _mm256_and_si256(m, _mm256_set1_epi8(1));
      };

    int x = xstart;
    for (; x + 32 <= xstop; x += 32)
    {
      // A: Center pixel > 3 check
      __m256i maskA = ld(gt3_dp, x);
      if (_mm256_testz_si256(maskA, maskA)) continue;
      // B: 3x3 neighbors > 3 check
      __m256i anyNbr3 = _mm256_or_si256(
        _mm256_or_si256(
          _mm256_or_si256(ld(gt3_dpp, x - 1), ld(gt3_dpp, x)),
          _mm256_or_si256(ld(gt3_dpp, x + 1), ld(gt3_dp, x - 1))),
        _mm256_or_si256(
          _mm256_or_si256(ld(gt3_dp, x + 1), ld(gt3_dpn, x - 1)),
          _mm256_or_si256(ld(gt3_dpn, x), ld(gt3_dpn, x + 1))));

      __m256i active = _mm256_and_si256(maskA, anyNbr3);
      if (_mm256_testz_si256(active, active)) continue;
      // C: Base increment (+1)
      __m256i inc = _mm256_and_si256(active, one);
      // D: dp[x] > 19 check
      __m256i center19 = ld(gt19_dp, x);
      __m256i activeD = _mm256_and_si256(active, center19);
      if (_mm256_testz_si256(activeD, activeD)) {
        _mm256_storeu_si256((__m256i*)(dstp + x), _mm256_adds_epu8(ld(dstp, x), inc));
        continue;
      }
      // E: Count neighbors > 19 (edi/esi logic)
      __m256i dpp_m1 = ld(gt19_dpp, x - 1), dpp_0 = ld(gt19_dpp, x), dpp_p1 = ld(gt19_dpp, x + 1);
      __m256i dpn_m1 = ld(gt19_dpn, x - 1), dpn_0 = ld(gt19_dpn, x), dpn_p1 = ld(gt19_dpn, x + 1);
      __m256i dp_m1 = ld(gt19_dp, x - 1), dp_p1 = ld(gt19_dp, x + 1);

      __m256i upper_quick = _mm256_or_si256(_mm256_or_si256(dpp_m1, dpp_0), dpp_p1);
      __m256i lower_quick = _mm256_or_si256(_mm256_or_si256(dpn_m1, dpn_0), dpn_p1);

      __m256i edi = _mm256_add_epi8(_mm256_add_epi8(b1(dpp_m1), b1(dpp_0)), b1(dpp_p1));
      __m256i esi_cnt = _mm256_add_epi8(edi, _mm256_add_epi8(b1(dp_m1), b1(dp_p1)));
      edi = _mm256_add_epi8(esi_cnt,
        _mm256_add_epi8(_mm256_add_epi8(b1(dpn_m1), b1(dpn_0)), b1(dpn_p1)));

      __m256i activeE = _mm256_and_si256(activeD, _mm256_cmpgt_epi8(edi, two));
      if (_mm256_testz_si256(activeE, activeE)) {
        _mm256_storeu_si256((__m256i*)(dstp + x), _mm256_adds_epu8(ld(dstp, x), inc));
        continue;
      }
      // Quick path: both upper and lower 3x3 ring neighbors exist
      // scalar does dstp[x]++ then +=2, total +3; inc has +1, add 2 more.
      __m256i quickPath = _mm256_and_si256(activeE, _mm256_and_si256(lower_quick, upper_quick));
      inc = _mm256_adds_epu8(inc, _mm256_and_si256(quickPath, two));

      // Only compute wide horzOR9 scan if pixels remain that need it
      __m256i needScan = _mm256_andnot_si256(quickPath, activeE);
      if (!_mm256_testz_si256(needScan, needScan))
      {
        // F: Wide +/-4 scan — horzOR9 in xmm, merged to ymm via set_m128i
        __m256i upper_w = horzOR9_x32(gt19_dpp, x);
        __m256i lower_w = horzOR9_x32(gt19_dpn, x);
        __m256i upper2_w = (y != 2) ? horzOR9_x32(gt19_dppp, x) : z256;
        __m256i lower2_w = (y != Height - 4) ? horzOR9_x32(gt19_dpnn, x) : z256;

        __m256i edi_gt4 = _mm256_cmpgt_epi8(edi, c4);
        __m256i upper_zero = _mm256_cmpeq_epi8(upper_w, z256);
        __m256i upper_nz = _mm256_andnot_si256(upper_zero, ff256);
        __m256i lower_zero = _mm256_cmpeq_epi8(lower_w, z256);
        __m256i lower2_zero = _mm256_cmpeq_epi8(lower2_w, z256);
        __m256i lower_nz = _mm256_andnot_si256(lower_zero, ff256);
        __m256i upper2_nz = _mm256_andnot_si256(_mm256_cmpeq_epi8(upper2_w, z256), ff256);

        __m256i lo_or_lo2_zero = _mm256_or_si256(lower_zero, lower2_zero);
        __m256i add4_u0 = _mm256_and_si256(_mm256_and_si256(upper_zero, lo_or_lo2_zero), edi_gt4);
        __m256i add2_u0 = _mm256_and_si256(upper_zero, _mm256_andnot_si256(lo_or_lo2_zero, ff256));
        __m256i lo_or_u2 = _mm256_or_si256(lower_nz, upper2_nz);
        __m256i add2_u1 = _mm256_and_si256(upper_nz, lo_or_u2);
        __m256i add4_u1 = _mm256_and_si256(_mm256_and_si256(upper_nz,
          _mm256_andnot_si256(lo_or_u2, ff256)), edi_gt4);

        __m256i add2 = _mm256_and_si256(_mm256_or_si256(add2_u0, add2_u1), needScan);
        __m256i add4 = _mm256_and_si256(_mm256_or_si256(add4_u0, add4_u1), needScan);

        // Wide scan outcomes — inc already holds +1 (base); add on top:
        // +3 total (scalar's dstp[x]++ + +=2): add 2 more
        inc = _mm256_adds_epu8(inc, _mm256_and_si256(add2, two));
        // +5 total (scalar's dstp[x]++ + +=4): add 4 more
        inc = _mm256_adds_epu8(inc, _mm256_and_si256(add4, _mm256_set1_epi8(4)));
      }
      _mm256_storeu_si256((__m256i*)(dstp + x), _mm256_adds_epu8(ld(dstp, x), inc));
    }
    // Remainder (< 32 pixels) via SSE4.1
    if (x < xstop) {
      ProcessRow_SSE41(gt3_dppp, gt19_dppp, gt3_dpp, gt19_dpp,
        gt3_dp, gt19_dp, gt3_dpn, gt19_dpn,
        gt3_dpnn, gt19_dpnn,
        dstp, x, xstop, y, Width, Height);
    }
  }
#endif // AVX2
  // ─────────────────────────────────────────────────────────────────────────────
  // Boolean row ring buffer: 5 slots, pointer-rotated each y iteration
  // Only the new dpnn row is rebuilt per step; the other 4 are just pointer moves.
  // ─────────────────────────────────────────────────────────────────────────────

  struct BoolRowRing {
    int C3, C19;
    int stride;
    uint8_t* mem;
    uint8_t* gt3[5]; // dppp, dpp, dp, dpn, dpnn
    uint8_t* gt19[5];

    BoolRowRing(int bits_per_pixel) : C3(3 << (bits_per_pixel - 8)), C19(19 << (bits_per_pixel - 8)), stride(0), mem(nullptr) {};

    void alloc(int width) {
      stride = (BROW_PAD + width + BROW_PAD + 63) & ~63;
      mem = static_cast<uint8_t*>(_aligned_malloc(10 * stride, 64));
      // Zero the entire allocation once.  BuildBoolRow_u8/16 only writes to the
      // active [0..width-1] region, so every padding byte stays zero for the
      // lifetime of the ring buffer — including after pointer rotation.
      memset(mem, 0, 10 * stride);
      for (int i = 0; i < 5; i++) {
        gt3[i] = mem + (2 * i) * stride + BROW_PAD;
        gt19[i] = mem + (2 * i + 1) * stride + BROW_PAD;
      }
    }
    void free_mem() { _aligned_free(mem); }

    void zero(int slot) {
      // gt3 and gt19 for this slot are already zeroed from alloc().
      // Just memset the active region to be safe after any previous rotation.
      memset(gt3[slot] - BROW_PAD, 0, stride);
      memset(gt19[slot] - BROW_PAD, 0, stride);
    }

    template<typename pixel_t>
    void build(int slot, const pixel_t* src, int width) {
      if constexpr (sizeof(pixel_t) == 1) {
        BuildBoolRow_u8(src, width, gt3[slot], gt19[slot], C3, C19);
      }
      else {
        BuildBoolRow_u16(src, width, gt3[slot], gt19[slot], C3, C19);
      }
    }

    template<typename pixel_t>
    void rotate(const pixel_t* new_dpnn, int width) {
      uint8_t* old3 = gt3[0];
      uint8_t* old19 = gt19[0];
      for (int i = 0; i < 4; i++) { gt3[i] = gt3[i + 1]; gt19[i] = gt19[i + 1]; }
      gt3[4] = old3;
      gt19[4] = old19;
      build<pixel_t>(4, new_dpnn, width);
    }
  };

  template<typename pixel_t>
  void AnalyzeDiffMask_Planar_C
  (
    uint8_t* dstp, int dst_pitch,
    uint8_t* tbuffer8, int tpitch,
    int Width, int Height, int bits_per_pixel)
  {
    if constexpr (sizeof(pixel_t) == 1)
      bits_per_pixel = 8; // quasi constexpr
    const int C3 = 3 << (bits_per_pixel - 8);
    const int C19 = 19 << (bits_per_pixel - 8);
    tpitch /= sizeof(pixel_t);
    const pixel_t* tbuffer = reinterpret_cast<const pixel_t*>(tbuffer8);
    const pixel_t* dppp = tbuffer - tpitch;
    const pixel_t* dpp = tbuffer;
    const pixel_t* dp = tbuffer + tpitch;
    const pixel_t* dpn = tbuffer + tpitch * 2;
    const pixel_t* dpnn = tbuffer + tpitch * 3;

    for (int y = 2; y < Height - 2; y += 2) {
      for (int x = 1; x < Width - 1; x++)
        AnalyzeOnePixel_D1<pixel_t>(
          dstp, dppp, dpp, dp, dpn, dpnn, x, y, Width, Height, C3, C19);
      dppp += tpitch; dpp += tpitch; dp += tpitch;
      dpn += tpitch; dpnn += tpitch;
      dstp += dst_pitch;
    }
  }

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point — drop-in for AnalyzeDiffMask_Planar<pixel_t, bpp>
// ─────────────────────────────────────────────────────────────────────────────
template<typename pixel_t>
#ifdef INCLUDE_PROCESSROW_AVX2
void AnalyzeDiffMask_Planar_AVX2
#else
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif 
void AnalyzeDiffMask_Planar_SSE41
#endif
(
  uint8_t* dstp, int dst_pitch,
  uint8_t* tbuffer8, int tpitch,
  int Width, int Height, int bits_per_pixel)
{
  tpitch /= sizeof(pixel_t);
  const pixel_t* tbuffer = reinterpret_cast<const pixel_t*>(tbuffer8);
  const pixel_t* dppp = tbuffer - tpitch;
  const pixel_t* dpp = tbuffer;
  const pixel_t* dp = tbuffer + tpitch;
  const pixel_t* dpn = tbuffer + tpitch * 2;
  const pixel_t* dpnn = tbuffer + tpitch * 3;

  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // quasi constexpr

  const int C3 = 3 << (bits_per_pixel - 8);
  const int C19 = 19 << (bits_per_pixel - 8);

#if 0
  // C only, kept for reference
  for (int y = 2; y < Height - 2; y += 2) {
    for (int x = 1; x < Width - 1; x++)
      AnalyzeOnePixel_D1<pixel_t>(
        dstp, dppp, dpp, dp, dpn, dpnn, x, y, Width, Height, C3, C19);
    dppp += tpitch; dpp += tpitch; dp += tpitch;
    dpn += tpitch; dpnn += tpitch;
    dstp += dst_pitch;
  }
  return;
#endif

  // SIMD body covers x in [xsimd_start, xsimd_stop_aligned).
  // xsimd_stop_aligned is rounded DOWN to the SIMD block size so the
  // right-edge scalar loop below covers [xsimd_stop_aligned..Width-1)
  // with no gap. Without this, up to 15 pixels per row near the right
  // edge (e.g. x in [1909..1915) for Width=1920) are silently skipped,
  // causing SIMD output to diverge from scalar.

  // Use 16-alignment for BOTH to let AVX2 handle the 16-pixel sub-remainder
  // AVX2 loop is 32-wide; its SSE41 sub-remainder is 16-wide.
  const int xsimd_start = 5;
  const int xsimd_stop_raw = Width - 5;
  const int xsimd_stop = xsimd_start + ((xsimd_stop_raw - xsimd_start) & ~15);

  BoolRowRing brows(bits_per_pixel);
  brows.alloc(Width);

  // dppp not yet valid at y==2; zero slot 0 instead of reading before tbuffer.
  // By the time y advances past 2 and dppp is needed, rotate() will have
  // filled slot 0 with valid data from what was dpnn two iterations ago.
  brows.zero(0);
  brows.build<pixel_t>(1, dpp, Width);
  brows.build<pixel_t>(2, dp, Width);
  brows.build<pixel_t>(3, dpn, Width);
  brows.build<pixel_t>(4, dpnn, Width);

  for (int y = 2; y < Height - 2; y += 2)
  {
    // Left edge — scalar (x = 1 .. xsimd_start-1)
    for (int x = 1; x < std::min(xsimd_start, Width - 1); x++)
      AnalyzeOnePixel_D1<pixel_t>(dstp,
        dppp, dpp, dp, dpn, dpnn,
        x, y, Width, Height, C3, C19);

    // SIMD body
    if (xsimd_stop > xsimd_start) {

#ifdef INCLUDE_PROCESSROW_AVX2
      ProcessRow_AVX2(
#else
      ProcessRow_SSE41(
#endif
        brows.gt3[0], brows.gt19[0],
        brows.gt3[1], brows.gt19[1],
        brows.gt3[2], brows.gt19[2],
        brows.gt3[3], brows.gt19[3],
        brows.gt3[4], brows.gt19[4],
        dstp, xsimd_start, xsimd_stop, y, Width, Height);
    }

    // Right edge — scalar (x = xsimd_stop .. Width-2)
    for (int x = std::max(xsimd_stop, xsimd_start); x < Width - 1; x++)
      AnalyzeOnePixel_D1<pixel_t>(dstp,
        dppp, dpp, dp, dpn, dpnn,
        x, y, Width, Height, C3, C19);

    // Advance source row pointers
    dppp += tpitch; dpp += tpitch; dp += tpitch;
    dpn += tpitch; dpnn += tpitch;

    // Only rotate if there is a next iteration to serve
    // Rotate ring: rebuild only the new dpnn boolean row
    if (y + 2 < Height - 2)
      brows.rotate<pixel_t>(dpnn, Width);

    dstp += dst_pitch;
  }

  brows.free_mem();
}
