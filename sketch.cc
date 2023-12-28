#include <cassert>
#include <cstdint>
#include <type_traits>
#include <limits>

#include <stdexcept>

#include <riscv_vector.h>

namespace rvv {

enum LMUL {
    LMUL_mf8 = -3,
    LMUL_mf4 = -2,
    LMUL_mf2 = -1,
    LMUL_m1 = 0,
    LMUL_m2 = 1,
    LMUL_m4 = 2,
    LMUL_m8 = 3,
};

namespace impl {
    /*using float8_t = _Float8;*/
    using float16_t = _Float16;

    // Use the nearest fit for unsupported part-register types
    //
    using  vuint64mf2_t =  vuint64m1_t;
    using   vint64mf2_t =   vint64m1_t;
    using vfloat64mf2_t = vfloat64m1_t;
    using  vuint64mf4_t =  vuint64m1_t;
    using   vint64mf4_t =   vint64m1_t;
    using vfloat64mf4_t = vfloat64m1_t;
    using  vuint32mf4_t = vuint32mf2_t;
    using   vint32mf4_t =  vint32mf2_t;
    using vfloat32mf4_t =vfloat32mf2_t;
    using  vuint64mf8_t =  vuint64m1_t;
    using   vint64mf8_t =   vint64m1_t;
    using vfloat64mf8_t = vfloat64m1_t;
    using  vuint32mf8_t = vuint32mf2_t;
    using   vint32mf8_t =  vint32mf2_t;
    using vfloat32mf8_t =vfloat32mf2_t;
    using  vuint16mf8_t = vuint16mf4_t;
    using   vint16mf8_t =  vint16mf4_t;

#define  TC_XMACRO8(X, M)                   X( u8##M, 8,M) X( i8##M, 8,M)
#if defined __riscv_zvfh
#define TC_XMACRO16(X, M)  TC_XMACRO8(X, M) X(u16##M,16,M) X(i16##M,16,M) X(f16##M,16,M)
#else
#define TC_XMACRO16(X, M)  TC_XMACRO8(X, M) X(u16##M,16,M) X(i16##M,16,M)
#endif
#define TC_XMACRO32(X, M) TC_XMACRO16(X, M) X(u32##M,32,M) X(i32##M,32,M) X(f32##M,32,M)
#define TC_XMACRO64(X, M) TC_XMACRO32(X, M) X(u64##M,64,M) X(i64##M,64,M) X(f64##M,64,M)
#define TC_XMACRO(X) TC_XMACRO8(X, mf8) TC_XMACRO16(X, mf4) TC_XMACRO32(X, mf2) TC_XMACRO64(X, m1) TC_XMACRO64(X, m2) TC_XMACRO64(X, m4) TC_XMACRO64(X, m8)
#define TC_XMACRO_SQUARE(X) TC_XMACRO64(X, mf8) TC_XMACRO64(X, mf4) TC_XMACRO64(X, mf2) TC_XMACRO64(X, m1) TC_XMACRO64(X, m2) TC_XMACRO64(X, m4) TC_XMACRO64(X, m8)

#define LMUL_XMACRO(X) X(mf8) X(mf4) X(mf2) X(m1) X(m2) X(m4) X(m8)

#define VECTOR_CAT(m) \
        using  u8##m =   vuint8##m##_t;  \
        using  i8##m =    vint8##m##_t;  \
      /*using  f8##m =  vfloat8##m##_t;*/\
        using u16##m =  vuint16##m##_t;  \
        using i16##m =   vint16##m##_t;  \
        using f16##m = vfloat16##m##_t;  \
        using u32##m =  vuint32##m##_t;  \
        using i32##m =   vint32##m##_t;  \
        using f32##m = vfloat32##m##_t;  \
        using u64##m =  vuint64##m##_t;  \
        using i64##m =   vint64##m##_t;  \
        using f64##m = vfloat64##m##_t;
#define SCALAR_CAT(m) \
        using  u8##m =    uint8_t;  \
        using  i8##m =     int8_t;  \
      /*using  f8##m =   float8_t;*/\
        using u16##m =   uint16_t;  \
        using i16##m =    int16_t;  \
        using f16##m =  float16_t;  \
        using u32##m =   uint32_t;  \
        using i32##m =    int32_t;  \
        using f32##m =      float;  \
        using u64##m =   uint64_t;  \
        using i64##m =    int64_t;  \
        using f64##m =     double;
#define LMUL_CAT(m) \
        static constexpr LMUL  u8##m = LMUL_##m;  \
        static constexpr LMUL  i8##m = LMUL_##m;  \
      /*static constexpr LMUL  f8##m = LMUL_##m;*/\
        static constexpr LMUL u16##m = LMUL_##m;  \
        static constexpr LMUL i16##m = LMUL_##m;  \
        static constexpr LMUL f16##m = LMUL_##m;  \
        static constexpr LMUL u32##m = LMUL_##m;  \
        static constexpr LMUL i32##m = LMUL_##m;  \
        static constexpr LMUL f32##m = LMUL_##m;  \
        static constexpr LMUL u64##m = LMUL_##m;  \
        static constexpr LMUL i64##m = LMUL_##m;  \
        static constexpr LMUL f64##m = LMUL_##m;

    struct tc {
        template <typename T, LMUL m>
        struct unsupported_type;
#if !defined __riscv_zvfh
        using vfloat16mf8_t = unsupported_type<float16_t, LMUL_mf8>;
        using vfloat16mf4_t = unsupported_type<float16_t, LMUL_mf4>;
        using vfloat16mf2_t = unsupported_type<float16_t, LMUL_mf2>;
        using vfloat16m1_t  = unsupported_type<float16_t, LMUL_m1>;
        using vfloat16m2_t  = unsupported_type<float16_t, LMUL_m2>;
        using vfloat16m4_t  = unsupported_type<float16_t, LMUL_m4>;
        using vfloat16m8_t  = unsupported_type<float16_t, LMUL_m8>;
#endif

        struct v { LMUL_XMACRO(VECTOR_CAT) };
        struct s { LMUL_XMACRO(SCALAR_CAT) };
        struct m { LMUL_XMACRO(LMUL_CAT) };
    };
    template <typename T>
    inline constexpr bool is_fundamental_v = std::is_fundamental_v<T> || std::is_same_v<T, float16_t>;

#undef SCALAR_CAT
#undef VECTOR_CAT
#undef LMUL_CAT

    template <typename T, LMUL m /*, std::enable_if_t<std::is_fundamental_v<T>, bool> = true*/ >
    struct rv_meta_base : public std::numeric_limits<T> {
        static_assert(is_fundamental_v<T>, "non-fundamental type passed to rv_meta<>");
        using lane_type = T;
        static constexpr LMUL lmul = m;
        static constexpr std::size_t min_lanes = (std::size_t(__riscv_v_min_vlen / 8) << (int(lmul) + 3) >> 3) / sizeof(lane_type);
#if defined __riscv_v_fixed_vlen
        static constexpr std::size_t num_lanes = (std::size_t(__riscv_v_fixed_vlen / 8) << (int(lmul) + 3) >> 3) / sizeof(lane_type);
#endif

        static constexpr bool can_ext = (lmul != LMUL_m8);
        static constexpr bool can_trunc = (lmul != LMUL_mf8);
        static constexpr bool can_widen = sizeof(lane_type) <= 4;
        static constexpr bool can_narrow = sizeof(lane_type) >= 2;
    };
    template <typename T, LMUL m> struct rv_meta;

    template <typename T> struct widen;
    template<> struct widen<uint8_t>  { using type = uint16_t; };
    template<> struct widen<uint16_t> { using type = uint32_t; };
    template<> struct widen<uint32_t> { using type = uint64_t; };
    template<> struct widen<int8_t>   { using type = int16_t; };
    template<> struct widen<int16_t>  { using type = int32_t; };
    template<> struct widen<int32_t>  { using type = int64_t; };
    template<> struct widen<float16_t>{ using type = float; };
    template<> struct widen<float>    { using type = double; };

    template <typename T> struct narrow;
    template<> struct narrow<uint16_t> { using type = uint8_t; };
    template<> struct narrow<uint32_t> { using type = uint16_t; };
    template<> struct narrow<uint64_t> { using type = uint32_t; };
    template<> struct narrow<int16_t>  { using type = int8_t; };
    template<> struct narrow<int32_t>  { using type = int16_t; };
    template<> struct narrow<int64_t>  { using type = int32_t; };
    template<> struct narrow<float>    { using type = float16_t; };
    template<> struct narrow<double>   { using type = float; };

    inline constexpr LMUL widen_lmul(LMUL x) {
        return int(x) < int(LMUL_m8) ? LMUL(int(x) + 1) : (throw "out-of-bounds lmul");
    }

    inline constexpr LMUL narrow_lmul(LMUL x) {
        return int(x) > int(LMUL_mf8) ? LMUL(int(x) - 1) : (throw "out-of-bounds lmul");
    }

    template <typename T, LMUL m>
    struct widen<rv_meta<T, m>> { using type = rv_meta<typename widen<typename rv_meta<T, m>::lane_type>::type, widen_lmul(rv_meta<T, m>::lmul)>; };
    template <typename T, LMUL m>
    struct narrow<rv_meta<T, m> > { using type = rv_meta<typename narrow<typename rv_meta<T, m>::lane_type>::type, narrow_lmul(rv_meta<T, m>::lmul)>; };

#define RV_META(T,S,M) \
        template <> struct rv_meta<tc::s::T, tc::m::T> : public rv_meta_base<tc::s::T, tc::m::T> { using reg_type = tc::v::T; };
#define RV_META_REV(T,S,M) \
        template <> struct rv_meta<tc::v::T,  LMUL_m1> : public rv_meta<tc::s::T, tc::m::T> {};
    TC_XMACRO_SQUARE(RV_META)
    TC_XMACRO(RV_META_REV)
#undef RV_META
#undef RV_META_REV

}  // namespace impl

// utilities

template <typename T, LMUL m = LMUL_m1> using rv_lane_t = typename impl::rv_meta<std::remove_reference_t<T>, m>::lane_type;
template <typename T, LMUL m = LMUL_m1> inline constexpr LMUL rv_lmul_v = impl::rv_meta<std::remove_reference_t<T>, m>::lmul;

// Using rv_lane_t and rv_lmul_v here ensures that rv_meta arguments are
// dereferenced through vector register type overloads:
template <typename T, LMUL m = LMUL_m1> using rv_meta = impl::rv_meta<rv_lane_t<T, m>, rv_lmul_v<T, m>>;

template <typename T> using widen_t = typename impl::widen<std::remove_reference_t<T>>::type;
template <typename T> using narrow_t = typename impl::narrow<std::remove_reference_t<T>>::type;
template <typename T, LMUL m = LMUL_m1> using rv_reg_t = typename rv_meta<T, m>::reg_type;
template <typename T, LMUL m = LMUL_m1> using rv_widen_reg_t = typename widen_t<rv_meta<T, m> >::reg_type;
template <typename T, LMUL m = LMUL_m1> using rv_narrow_reg_t = typename narrow_t<rv_meta<T, m> >::reg_type;

// Encoding the data type and LMUL in the type of the VL argument makes more
// type deduction feasible in places where there's no register input.
//
namespace impl {
    template <typename T>
    struct VLType {
        std::size_t vl_;
        VLType(std::size_t x) : vl_(x) {}
        operator std::size_t() const { return vl_; }

        using lane_type = typename T::lane_type;
        static constexpr LMUL lmul = T::lmul;

        // Wrap these in templates to ensure that illegal cases only give errors on
        // use rather than definition
        template <typename TT = T> VLType<widen_t<TT> > w() { return vl_; }
        template <typename TT = T> VLType<narrow_t<TT> > n() { return vl_; }
    };
}  // namespace impl

template <typename T, LMUL m = LMUL_m1> using VLType = impl::VLType<rv_meta<T, m> >;

#define RV_VLTYPE(T,S,M) using VLType_##T = VLType<impl::tc::s::T, impl::tc::m::T>;
TC_XMACRO_SQUARE(RV_VLTYPE)
#undef RV_VLTYPE


template <typename T, LMUL m = LMUL_m1>
inline VLType<T, m> rv_setvl(std::size_t avl) {
    static_assert(rv_meta<T, m>::min_lanes >= 1, "can't set LMUL that low");
    if (avl > 65536) avl = 65536;  // to avoid overflow
    static constexpr std::size_t sz = sizeof(T);
    switch (m) {
    case LMUL_mf8: return __riscv_vsetvl_e8mf8(avl * sz) / sz;
    case LMUL_mf4: return __riscv_vsetvl_e8mf4(avl * sz) / sz;
    case LMUL_mf2: return __riscv_vsetvl_e8mf2(avl * sz) / sz;
    case LMUL_m1: return __riscv_vsetvl_e8m1(avl * sz) / sz;
    case LMUL_m2: return __riscv_vsetvl_e8m2(avl * sz) / sz;
    case LMUL_m4: return __riscv_vsetvl_e8m4(avl * sz) / sz;
    case LMUL_m8: return __riscv_vsetvl_e8m8(avl * sz) / sz;
    }
}

#define RV_SETVL(T,S,M) \
        template <> VLType<impl::tc::s::T, impl::tc::m::T> rv_setvl<impl::tc::s::T, impl::tc::m::T>(std::size_t avl) { return __riscv_vsetvl_e##S##M(avl); }
TC_XMACRO(RV_SETVL)
#undef RV_SETVL

#if 0
inline VLType_u8mf2 rv_vsetvl_u8mf2(std::size_t avl) { return rv_setvl<VLType_u8mf2>(avl); }
inline VLType_i32mf2 rv_vsetvl_i32mf2(std::size_t avl) { return rv_setvl<VLType_i32mf2>(avl); }
inline VLType_u8m1 rv_vsetvl_u8m1(std::size_t avl) { return rv_setvl<VLType_u8m1>(avl); }
inline VLType_i32m1 rv_vsetvl_i32m1(std::size_t avl) { return rv_setvl<VLType_i32m1>(avl); }
inline VLType_u8m4 rv_vsetvl_u8m4(std::size_t avl) { return rv_setvl<VLType_u8m4>(avl); }
inline VLType_i32m4 rv_vsetvl_i32m4(std::size_t avl) { return rv_setvl<VLType_i32m4>(avl); }
#else
#define RV_SETVL_MORE(T,S,M) template <typename U = void>  /* not always legal, so template it */  \
        inline VLType_##T rv_setvl_##T(std::size_t avl) { return rv_setvl<impl::tc::s::T, impl::tc::m::T>(avl); }
TC_XMACRO_SQUARE(RV_SETVL_MORE)
#undef RV_SETVL_MORE
#endif


// Operations like vmv.vx and vle cannot deduce the output type from
// any of the arguments, unless it's coded in the vl type.
//
#define __riscv_vmv_v_x_f16mf4 __riscv_vfmv_v_f_f16mf4
#define __riscv_vmv_v_x_f16mf2 __riscv_vfmv_v_f_f16mf2
#define __riscv_vmv_v_x_f16m1  __riscv_vfmv_v_f_f16m1
#define __riscv_vmv_v_x_f16m2  __riscv_vfmv_v_f_f16m2
#define __riscv_vmv_v_x_f16m4  __riscv_vfmv_v_f_f16m4
#define __riscv_vmv_v_x_f16m8  __riscv_vfmv_v_f_f16m8
#define __riscv_vmv_v_x_f32mf2 __riscv_vfmv_v_f_f32mf2
#define __riscv_vmv_v_x_f32m1  __riscv_vfmv_v_f_f32m1
#define __riscv_vmv_v_x_f32m2  __riscv_vfmv_v_f_f32m2
#define __riscv_vmv_v_x_f32m4  __riscv_vfmv_v_f_f32m4
#define __riscv_vmv_v_x_f32m8  __riscv_vfmv_v_f_f32m8
#define __riscv_vmv_v_x_f64m1  __riscv_vfmv_v_f_f64m1
#define __riscv_vmv_v_x_f64m2  __riscv_vfmv_v_f_f64m2
#define __riscv_vmv_v_x_f64m4  __riscv_vfmv_v_f_f64m4
#define __riscv_vmv_v_x_f64m8  __riscv_vfmv_v_f_f64m8
#define RV_DUP(T,S,M) \
        inline impl::tc::v::T rv_dup(impl::tc::s::T v, VLType_##T vl) { return __riscv_vmv_v_x_##T(v, vl); }
TC_XMACRO(RV_DUP)
#undef RV_DUP
#undef __riscv_vmv_v_x_f16mf4
#undef __riscv_vmv_v_x_f16mf2
#undef __riscv_vmv_v_x_f16m1
#undef __riscv_vmv_v_x_f16m2
#undef __riscv_vmv_v_x_f16m4
#undef __riscv_vmv_v_x_f16m8
#undef __riscv_vmv_v_x_f32mf2
#undef __riscv_vmv_v_x_f32m1
#undef __riscv_vmv_v_x_f32m2
#undef __riscv_vmv_v_x_f32m4
#undef __riscv_vmv_v_x_f32m8
#undef __riscv_vmv_v_x_f64m1
#undef __riscv_vmv_v_x_f64m2
#undef __riscv_vmv_v_x_f64m4
#undef __riscv_vmv_v_x_f64m8


// Nothing special here.
//
template <typename Reg>
inline Reg rv_add(Reg x, Reg y, VLType<Reg> vl) { return __riscv_vadd(x, y, vl); }


// Instruction names change between signed and unsigned variants
// by necessity in assembly, but there's no such necessity in C.
//
namespace impl {
    template <bool is_integer, bool is_signed> struct rv_max {
        template <typename Reg> static inline Reg f(Reg x, Reg y, std::size_t vl) { return __riscv_vmax(x, y, vl); }
    };
    template <> struct rv_max<false, false> {
        template <typename Reg> static inline Reg f(Reg x, Reg y, std::size_t vl) { return __riscv_vfmax(x, y, vl); }
    };
    template <> struct rv_max<true, false> {
        template <typename Reg> static inline Reg f(Reg x, Reg y, std::size_t vl) { return __riscv_vmaxu(x, y, vl); }
    };
}  // namespace impl
template <typename Reg>
inline Reg rv_max(Reg x, Reg y, VLType<Reg> vl) {
    return impl::rv_max<rv_meta<Reg>::is_integer, rv_meta<Reg>::is_signed>::f(x, y, vl);
}

// and on it goes...
//
template <typename Reg>
inline rv_widen_reg_t<Reg> rv_wadd(Reg x, Reg y, VLType<Reg> vl) { return __riscv_vwadd_vv(x, y, vl); }

namespace impl {
    template <bool is_signed> struct rv_nshr {
        template <typename Reg> static inline rv_narrow_reg_t<Reg> f(Reg x, std::size_t i, std::size_t vl) { return __riscv_vnsra(x, i, vl); }
    };
    template <> struct rv_nshr<false> {
        template <typename Reg> static inline rv_narrow_reg_t<Reg> f(Reg x, std::size_t i, std::size_t vl) { return __riscv_vnsrl(x, i, vl); }
    };
}  // namespace impl
template <typename Reg>
inline rv_narrow_reg_t<Reg> rv_nshr(Reg x, std::size_t i, VLType<Reg> vl) {
    return impl::rv_nshr<rv_meta<Reg>::is_signed>::f(x, i, vl);
}


// __riscv_vle* and __riscv_vse* all include the specific bitwidth in
// the name, which undermines the utility of function overloading
//
#define RV_VLS(T,S,M) \
    inline impl::tc::v::T rv_vle(impl::tc::s::T const* p, VLType<impl::tc::s::T, impl::tc::m::T> vl) { return __riscv_vle##S##_v_##T(p, vl); }  \
    inline void rv_vse(impl::tc::s::T* p, impl::tc::v::T v, VLType<impl::tc::s::T, impl::tc::m::T> vl) { return __riscv_vse##S##_v_##T(p, v, vl); }
TC_XMACRO(RV_VLS)
#undef RV_VLS


// __riscv_vreinterpret_* overloads are not a complete set, and the names
// don't let you change things without getting in a muddle over other factors
//
inline vuint8mf2_t rv_reinterpret_u(vuint8mf2_t v) { return v; }
inline vuint8mf2_t rv_reinterpret_u(vint8mf2_t v) { return __riscv_vreinterpret_u8mf2(v); }
inline vuint8m1_t  rv_reinterpret_u(vuint8m1_t v) { return v; }
inline vuint8m1_t  rv_reinterpret_u(vint8m1_t v) { return __riscv_vreinterpret_u8m1(v); }
inline vuint8m4_t  rv_reinterpret_u(vuint8m4_t v) { return v; }
inline vuint8m4_t  rv_reinterpret_u(vint8m4_t v) { return __riscv_vreinterpret_u8m4(v); }
inline vuint32mf2_t rv_reinterpret_u(vuint32mf2_t v) { return v; }
inline vuint32mf2_t rv_reinterpret_u(vint32mf2_t v) { return __riscv_vreinterpret_u32mf2(v); }
inline vuint32m1_t rv_reinterpret_u(vuint32m1_t v) { return v; }
inline vuint32m1_t rv_reinterpret_u(vint32m1_t v) { return __riscv_vreinterpret_u32m1(v); }
inline vuint32m4_t rv_reinterpret_u(vuint32m4_t v) { return v; }
inline vuint32m4_t rv_reinterpret_u(vint32m4_t v) { return __riscv_vreinterpret_u32m4(v); }

inline vint8mf2_t rv_reinterpret_s(vint8mf2_t v) { return v; }
inline vint8mf2_t rv_reinterpret_s(vuint8mf2_t v) { return __riscv_vreinterpret_i8mf2(v); }
inline vint8m1_t  rv_reinterpret_s(vint8m1_t v) { return v; }
inline vint8m1_t  rv_reinterpret_s(vuint8m1_t v) { return __riscv_vreinterpret_i8m1(v); }
inline vint8m4_t  rv_reinterpret_s(vint8m4_t v) { return v; }
inline vint8m4_t  rv_reinterpret_s(vuint8m4_t v) { return __riscv_vreinterpret_i8m4(v); }
inline vint32mf2_t rv_reinterpret_s(vint32mf2_t v) { return v; }
inline vint32mf2_t rv_reinterpret_s(vuint32mf2_t v) { return __riscv_vreinterpret_i32mf2(v); }
inline vint32m1_t rv_reinterpret_s(vint32m1_t v) { return v; }
inline vint32m1_t rv_reinterpret_s(vuint32m1_t v) { return __riscv_vreinterpret_i32m1(v); }
inline vint32m4_t rv_reinterpret_s(vint32m4_t v) { return v; }
inline vint32m4_t rv_reinterpret_s(vuint32m4_t v) { return __riscv_vreinterpret_i32m4(v); }

inline vuint8mf2_t rv_reinterpret_8(vuint8mf2_t v) { return v; }
inline vuint8mf2_t rv_reinterpret_8(vuint32mf2_t v) { return __riscv_vreinterpret_u8mf2(v); }
inline vint8mf2_t  rv_reinterpret_8(vint8mf2_t v) { return v; }
inline vint8mf2_t  rv_reinterpret_8(vint32mf2_t v) { return __riscv_vreinterpret_i8mf2(v); }
inline vuint8m1_t rv_reinterpret_8(vuint8m1_t v) { return v; }
inline vuint8m1_t rv_reinterpret_8(vuint32m1_t v) { return __riscv_vreinterpret_u8m1(v); }
inline vint8m1_t  rv_reinterpret_8(vint8m1_t v) { return v; }
inline vint8m1_t  rv_reinterpret_8(vint32m1_t v) { return __riscv_vreinterpret_i8m1(v); }
inline vuint8m4_t rv_reinterpret_8(vuint8m4_t v) { return v; }
inline vuint8m4_t rv_reinterpret_8(vuint32m4_t v) { return __riscv_vreinterpret_u8m4(v); }
inline vint8m4_t  rv_reinterpret_8(vint8m4_t v) { return v; }
inline vint8m4_t  rv_reinterpret_8(vint32m4_t v) { return __riscv_vreinterpret_i8m4(v); }

inline vint8m1_t  rv_reinterpret_8(vint64m1_t v) { return __riscv_vreinterpret_i8m1(v); }
inline vuint8m1_t rv_reinterpret_8(vuint64m1_t v) { return __riscv_vreinterpret_u8m1(v); }

inline vuint32mf2_t rv_reinterpret_32(vuint32mf2_t v) { return v; }
inline vuint32mf2_t rv_reinterpret_32(vuint8mf2_t v) { return __riscv_vreinterpret_u32mf2(v); }
inline vint32mf2_t  rv_reinterpret_32(vint32mf2_t v) { return v; }
inline vint32mf2_t  rv_reinterpret_32(vint8mf2_t v) { return __riscv_vreinterpret_i32mf2(v); }
inline vuint32m1_t rv_reinterpret_32(vuint32m1_t v) { return v; }
inline vuint32m1_t rv_reinterpret_32(vuint8m1_t v) { return __riscv_vreinterpret_u32m1(v); }
inline vint32m1_t  rv_reinterpret_32(vint32m1_t v) { return v; }
inline vint32m1_t  rv_reinterpret_32(vint8m1_t v) { return __riscv_vreinterpret_i32m1(v); }
inline vuint32m4_t rv_reinterpret_32(vuint32m4_t v) { return v; }
inline vuint32m4_t rv_reinterpret_32(vuint8m4_t v) { return __riscv_vreinterpret_u32m4(v); }
inline vint32m4_t  rv_reinterpret_32(vint32m4_t v) { return v; }
inline vint32m4_t  rv_reinterpret_32(vint8m4_t v) { return __riscv_vreinterpret_i32m4(v); }

inline vuint64m1_t rv_reinterpret_64(vuint64m1_t v) { return v; }
inline vuint64m1_t rv_reinterpret_64(vuint8m1_t v) { return __riscv_vreinterpret_u64m1(v); }
inline vint64m1_t  rv_reinterpret_64(vint64m1_t v) { return v; }
inline vint64m1_t  rv_reinterpret_64(vint8m1_t v) { return __riscv_vreinterpret_i64m1(v); }
inline vuint64m4_t rv_reinterpret_64(vuint64m4_t v) { return v; }
inline vuint64m4_t rv_reinterpret_64(vuint8m4_t v) { return __riscv_vreinterpret_u64m4(v); }
inline vint64m4_t  rv_reinterpret_64(vint64m4_t v) { return v; }
inline vint64m4_t  rv_reinterpret_64(vint8m4_t v) { return __riscv_vreinterpret_i64m4(v); }

// When you use typedefs to make your LMUL configurable it becomes
// painful to cast to specific types while staying agnostic to LMUL.
//
template <typename T, typename U, std::enable_if_t<std::is_fundamental_v<T>, bool> = true>
struct rv_reinterpret_impl;
template <typename U>
struct rv_reinterpret_impl<uint8_t, U> { static decltype(auto) f(U v) { return rv_reinterpret_8(rv_reinterpret_u(v)); } };
template <typename U>
struct rv_reinterpret_impl<int8_t, U> { static decltype(auto) f(U v) { return rv_reinterpret_8(rv_reinterpret_s(v)); } };
template <typename U>
struct rv_reinterpret_impl<uint32_t, U> { static decltype(auto) f(U v) { return rv_reinterpret_32(rv_reinterpret_u(v)); } };
template <typename U>
struct rv_reinterpret_impl<int32_t, U> { static decltype(auto) f(U v) { return rv_reinterpret_32(rv_reinterpret_s(v)); } };
template <typename T, typename U>
inline decltype(auto) rv_reinterpret(U v) { return rv_reinterpret_impl<T, U>::f(v); }


template <typename T> inline decltype(auto)  rv_reinterpret_u8(T v) { return rv_reinterpret< uint8_t>(v); }
template <typename T> inline decltype(auto)  rv_reinterpret_s8(T v) { return rv_reinterpret<  int8_t>(v); }
template <typename T> inline decltype(auto) rv_reinterpret_u32(T v) { return rv_reinterpret<uint32_t>(v); }
template <typename T> inline decltype(auto) rv_reinterpret_s32(T v) { return rv_reinterpret< int32_t>(v); }


// ext and trunc macros without getting hung up in the peripheral details of the types
//
template <LMUL m_out, LMUL m_in>
struct rv_lmul_impl;

template <LMUL m_in>
struct rv_lmul_impl<LMUL_mf8, m_in> {
    template <typename T> static rv_reg_t<T, LMUL_mf8> t(rv_reg_t<T, m_in> v);
    template <> rv_reg_t<uint8_t, LMUL_mf8> t< uint8_t>(rv_reg_t<uint8_t, m_in> v) { return __riscv_vlmul_trunc_u8mf8(v); }
    template <> rv_reg_t< int8_t, LMUL_mf8> t<  int8_t>(rv_reg_t< int8_t, m_in> v) { return __riscv_vlmul_trunc_u8mf8(v); }
};

template <LMUL m_in>
struct rv_lmul_impl<LMUL_mf4, m_in> {
    template <typename T> static rv_reg_t<T, LMUL_mf4> e(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_mf4> e< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_ext_u8mf4(v); }
    template <> rv_reg_t<  int8_t, LMUL_mf4> e<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_ext_i8mf4(v); }
    template <> rv_reg_t<uint16_t, LMUL_mf4> e<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_ext_u16mf4(v); }
    template <> rv_reg_t< int16_t, LMUL_mf4> e< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_ext_i16mf4(v); }

    template <typename T> static rv_reg_t<T, LMUL_mf4> t(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_mf4> t< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_trunc_u8mf4(v); }
    template <> rv_reg_t<  int8_t, LMUL_mf4> t<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_trunc_i8mf4(v); }
    template <> rv_reg_t<uint16_t, LMUL_mf4> t<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_trunc_u16mf4(v); }
    template <> rv_reg_t< int16_t, LMUL_mf4> t< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_trunc_i16mf4(v); }
};

template <LMUL m_in>
struct rv_lmul_impl<LMUL_mf2, m_in> {
    template <typename T> static rv_reg_t<T, LMUL_mf2> e(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_mf2> e< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_ext_u8mf2(v); }
    template <> rv_reg_t<  int8_t, LMUL_mf2> e<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_ext_i8mf2(v); }
    template <> rv_reg_t<uint16_t, LMUL_mf2> e<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_ext_u16mf2(v); }
    template <> rv_reg_t< int16_t, LMUL_mf2> e< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_ext_i16mf2(v); }
    template <> rv_reg_t<uint32_t, LMUL_mf2> e<uint32_t>(rv_reg_t<uint32_t, m_in> v) { return __riscv_vlmul_ext_u32mf2(v); }
    template <> rv_reg_t< int32_t, LMUL_mf2> e< int32_t>(rv_reg_t< int32_t, m_in> v) { return __riscv_vlmul_ext_i32mf2(v); }

    template <typename T> static rv_reg_t<T, LMUL_mf2> t(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_mf2> t< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_trunc_u8mf2(v); }
    template <> rv_reg_t<  int8_t, LMUL_mf2> t<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_trunc_i8mf2(v); }
    template <> rv_reg_t<uint16_t, LMUL_mf2> t<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_trunc_u16mf2(v); }
    template <> rv_reg_t< int16_t, LMUL_mf2> t< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_trunc_i16mf2(v); }
    template <> rv_reg_t<uint32_t, LMUL_mf2> t<uint32_t>(rv_reg_t<uint32_t, m_in> v) { return __riscv_vlmul_trunc_u32mf2(v); }
    template <> rv_reg_t< int32_t, LMUL_mf2> t< int32_t>(rv_reg_t< int32_t, m_in> v) { return __riscv_vlmul_trunc_i32mf2(v); }
};

template <LMUL m_in>
struct rv_lmul_impl<LMUL_m1, m_in> {
    template <typename T> static rv_reg_t<T, LMUL_m1> e(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_m1> e< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_ext_u8m1(v); }
    template <> rv_reg_t<  int8_t, LMUL_m1> e<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_ext_i8m1(v); }
    template <> rv_reg_t<uint16_t, LMUL_m1> e<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_ext_u16m1(v); }
    template <> rv_reg_t< int16_t, LMUL_m1> e< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_ext_i16m1(v); }
    template <> rv_reg_t<uint32_t, LMUL_m1> e<uint32_t>(rv_reg_t<uint32_t, m_in> v) { return __riscv_vlmul_ext_u32m1(v); }
    template <> rv_reg_t< int32_t, LMUL_m1> e< int32_t>(rv_reg_t< int32_t, m_in> v) { return __riscv_vlmul_ext_i32m1(v); }
    template <> rv_reg_t<uint64_t, LMUL_m1> e<uint64_t>(rv_reg_t<uint64_t, m_in> v) { return __riscv_vlmul_ext_u64m1(v); }
    template <> rv_reg_t< int64_t, LMUL_m1> e< int64_t>(rv_reg_t< int64_t, m_in> v) { return __riscv_vlmul_ext_i64m1(v); }

    template <typename T> static rv_reg_t<T, LMUL_m1> t(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_m1> t< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_trunc_u8m1(v); }
    template <> rv_reg_t<  int8_t, LMUL_m1> t<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_trunc_i8m1(v); }
    template <> rv_reg_t<uint16_t, LMUL_m1> t<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_trunc_u16m1(v); }
    template <> rv_reg_t< int16_t, LMUL_m1> t< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_trunc_i16m1(v); }
    template <> rv_reg_t<uint32_t, LMUL_m1> t<uint32_t>(rv_reg_t<uint32_t, m_in> v) { return __riscv_vlmul_trunc_u32m1(v); }
    template <> rv_reg_t< int32_t, LMUL_m1> t< int32_t>(rv_reg_t< int32_t, m_in> v) { return __riscv_vlmul_trunc_i32m1(v); }
    template <> rv_reg_t<uint64_t, LMUL_m1> t<uint64_t>(rv_reg_t<uint64_t, m_in> v) { return __riscv_vlmul_trunc_u64m1(v); }
    template <> rv_reg_t< int64_t, LMUL_m1> t< int64_t>(rv_reg_t< int64_t, m_in> v) { return __riscv_vlmul_trunc_i64m1(v); }
};

template <LMUL m_in>
struct rv_lmul_impl<LMUL_m2, m_in> {
    template <typename T> static rv_reg_t<T, LMUL_m2> e(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_m2> e< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_ext_u8m2(v); }
    template <> rv_reg_t<  int8_t, LMUL_m2> e<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_ext_i8m2(v); }
    template <> rv_reg_t<uint16_t, LMUL_m2> e<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_ext_u16m2(v); }
    template <> rv_reg_t< int16_t, LMUL_m2> e< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_ext_i16m2(v); }
    template <> rv_reg_t<uint32_t, LMUL_m2> e<uint32_t>(rv_reg_t<uint32_t, m_in> v) { return __riscv_vlmul_ext_u32m2(v); }
    template <> rv_reg_t< int32_t, LMUL_m2> e< int32_t>(rv_reg_t< int32_t, m_in> v) { return __riscv_vlmul_ext_i32m2(v); }
    template <> rv_reg_t<uint64_t, LMUL_m2> e<uint64_t>(rv_reg_t<uint64_t, m_in> v) { return __riscv_vlmul_ext_u64m2(v); }
    template <> rv_reg_t< int64_t, LMUL_m2> e< int64_t>(rv_reg_t< int64_t, m_in> v) { return __riscv_vlmul_ext_i64m2(v); }

    template <typename T> static rv_reg_t<T, LMUL_m2> t(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_m2> t< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_trunc_u8m2(v); }
    template <> rv_reg_t<  int8_t, LMUL_m2> t<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_trunc_i8m2(v); }
    template <> rv_reg_t<uint16_t, LMUL_m2> t<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_trunc_u16m2(v); }
    template <> rv_reg_t< int16_t, LMUL_m2> t< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_trunc_i16m2(v); }
    template <> rv_reg_t<uint32_t, LMUL_m2> t<uint32_t>(rv_reg_t<uint32_t, m_in> v) { return __riscv_vlmul_trunc_u32m2(v); }
    template <> rv_reg_t< int32_t, LMUL_m2> t< int32_t>(rv_reg_t< int32_t, m_in> v) { return __riscv_vlmul_trunc_i32m2(v); }
    template <> rv_reg_t<uint64_t, LMUL_m2> t<uint64_t>(rv_reg_t<uint64_t, m_in> v) { return __riscv_vlmul_trunc_u64m2(v); }
    template <> rv_reg_t< int64_t, LMUL_m2> t< int64_t>(rv_reg_t< int64_t, m_in> v) { return __riscv_vlmul_trunc_i64m2(v); }
};

template <LMUL m_in>
struct rv_lmul_impl<LMUL_m4, m_in> {
    template <typename T> static rv_reg_t<T, LMUL_m4> e(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_m4> e< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_ext_u8m4(v); }
    template <> rv_reg_t<  int8_t, LMUL_m4> e<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_ext_i8m4(v); }
    template <> rv_reg_t<uint16_t, LMUL_m4> e<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_ext_u16m4(v); }
    template <> rv_reg_t< int16_t, LMUL_m4> e< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_ext_i16m4(v); }
    template <> rv_reg_t<uint32_t, LMUL_m4> e<uint32_t>(rv_reg_t<uint32_t, m_in> v) { return __riscv_vlmul_ext_u32m4(v); }
    template <> rv_reg_t< int32_t, LMUL_m4> e< int32_t>(rv_reg_t< int32_t, m_in> v) { return __riscv_vlmul_ext_i32m4(v); }
    template <> rv_reg_t<uint64_t, LMUL_m4> e<uint64_t>(rv_reg_t<uint64_t, m_in> v) { return __riscv_vlmul_ext_u64m4(v); }
    template <> rv_reg_t< int64_t, LMUL_m4> e< int64_t>(rv_reg_t< int64_t, m_in> v) { return __riscv_vlmul_ext_i64m4(v); }

    template <typename T> static rv_reg_t<T, LMUL_m4> t(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_m4> t< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_trunc_u8m4(v); }
    template <> rv_reg_t<  int8_t, LMUL_m4> t<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_trunc_i8m4(v); }
    template <> rv_reg_t<uint16_t, LMUL_m4> t<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_trunc_u16m4(v); }
    template <> rv_reg_t< int16_t, LMUL_m4> t< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_trunc_i16m4(v); }
    template <> rv_reg_t<uint32_t, LMUL_m4> t<uint32_t>(rv_reg_t<uint32_t, m_in> v) { return __riscv_vlmul_trunc_u32m4(v); }
    template <> rv_reg_t< int32_t, LMUL_m4> t< int32_t>(rv_reg_t< int32_t, m_in> v) { return __riscv_vlmul_trunc_i32m4(v); }
    template <> rv_reg_t<uint64_t, LMUL_m4> t<uint64_t>(rv_reg_t<uint64_t, m_in> v) { return __riscv_vlmul_trunc_u64m4(v); }
    template <> rv_reg_t< int64_t, LMUL_m4> t< int64_t>(rv_reg_t< int64_t, m_in> v) { return __riscv_vlmul_trunc_i64m4(v); }
};

template <LMUL m_in>
struct rv_lmul_impl<LMUL_m8, m_in> {
    template <typename T> static rv_reg_t<T, LMUL_m8> e(rv_reg_t<T, m_in> v);
    template <> rv_reg_t< uint8_t, LMUL_m8> e< uint8_t>(rv_reg_t< uint8_t, m_in> v) { return  __riscv_vlmul_ext_u8m8(v); }
    template <> rv_reg_t<  int8_t, LMUL_m8> e<  int8_t>(rv_reg_t<  int8_t, m_in> v) { return  __riscv_vlmul_ext_i8m8(v); }
    template <> rv_reg_t<uint16_t, LMUL_m8> e<uint16_t>(rv_reg_t<uint16_t, m_in> v) { return __riscv_vlmul_ext_u16m8(v); }
    template <> rv_reg_t< int16_t, LMUL_m8> e< int16_t>(rv_reg_t< int16_t, m_in> v) { return __riscv_vlmul_ext_i16m8(v); }
    template <> rv_reg_t<uint32_t, LMUL_m8> e<uint32_t>(rv_reg_t<uint32_t, m_in> v) { return __riscv_vlmul_ext_u32m8(v); }
    template <> rv_reg_t< int32_t, LMUL_m8> e< int32_t>(rv_reg_t< int32_t, m_in> v) { return __riscv_vlmul_ext_i32m8(v); }
    template <> rv_reg_t<uint64_t, LMUL_m8> e<uint64_t>(rv_reg_t<uint64_t, m_in> v) { return __riscv_vlmul_ext_u64m8(v); }
    template <> rv_reg_t< int64_t, LMUL_m8> e< int64_t>(rv_reg_t< int64_t, m_in> v) { return __riscv_vlmul_ext_i64m8(v); }
};

template <LMUL m_out, typename Reg, typename T = rv_lane_t<Reg>, LMUL m_in = rv_lmul_v<Reg>>
rv_reg_t<T, m_out> rv_lmul_ext(Reg v) { return rv_lmul_impl<m_out, m_in>::template e<T>(v); }
template <LMUL m_out, typename Reg, typename T = rv_lane_t<Reg>, LMUL m_in = rv_lmul_v<Reg>>
rv_reg_t<T, m_out> rv_lmul_trunc(Reg v) { return rv_lmul_impl<m_out, m_in>::template t<T>(v); }

// absolute versions
//
template <typename Reg> rv_reg_t<rv_lane_t<Reg>, LMUL_mf8> rv_lmul_ext_mf8(Reg v) { return rv_lmul_ext<LMUL_mf8>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>, LMUL_mf4> rv_lmul_ext_mf4(Reg v) { return rv_lmul_ext<LMUL_mf4>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>, LMUL_mf2> rv_lmul_ext_mf2(Reg v) { return rv_lmul_ext<LMUL_mf2>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>,  LMUL_m1>  rv_lmul_ext_m1(Reg v) { return rv_lmul_ext<LMUL_m1>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>,  LMUL_m2>  rv_lmul_ext_m2(Reg v) { return rv_lmul_ext<LMUL_m2>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>,  LMUL_m4>  rv_lmul_ext_m4(Reg v) { return rv_lmul_ext<LMUL_m4>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>,  LMUL_m8>  rv_lmul_ext_m8(Reg v) { return rv_lmul_ext<LMUL_m8>(v); }

template <typename Reg> rv_reg_t<rv_lane_t<Reg>, LMUL_mf8> rv_lmul_trunc_mf8(Reg v) { return rv_lmul_trunc<LMUL_mf8>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>, LMUL_mf4> rv_lmul_trunc_mf4(Reg v) { return rv_lmul_trunc<LMUL_mf4>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>, LMUL_mf2> rv_lmul_trunc_mf2(Reg v) { return rv_lmul_trunc<LMUL_mf2>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>,  LMUL_m1>  rv_lmul_trunc_m1(Reg v) { return rv_lmul_trunc<LMUL_m1>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>,  LMUL_m2>  rv_lmul_trunc_m2(Reg v) { return rv_lmul_trunc<LMUL_m2>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>,  LMUL_m4>  rv_lmul_trunc_m4(Reg v) { return rv_lmul_trunc<LMUL_m4>(v); }
template <typename Reg> rv_reg_t<rv_lane_t<Reg>,  LMUL_m8>  rv_lmul_trunc_m8(Reg v) { return rv_lmul_trunc<LMUL_m8>(v); }

// relative versions
//
template <typename Reg, typename T = rv_lane_t<Reg>, LMUL m = impl::narrow_lmul(rv_lmul_v<Reg>)>
rv_reg_t<T, m> rv_lmul_trunc2(Reg v) { return rv_lmul_trunc<m>(v); }

template <typename Reg, typename T = rv_lane_t<Reg>, LMUL m = impl::widen_lmul(rv_lmul_v<Reg>)>
rv_reg_t<T, m> rv_lmul_ext2(Reg v) { return rv_lmul_ext<m>(v); }


// I can see in the spec why types like this don't exist, but they're still
// feasible and useful.
//
inline impl::vuint64mf2_t rv_dup(uint64_t v, VLType_u64mf2 vl) {
    VLType<impl::vuint64mf2_t> vl_delegate(vl);
    return rv_dup(v, vl_delegate);
}
inline impl::vuint64mf2_t rv_add(impl::vuint64mf2_t x, impl::vuint64mf2_t y, VLType_u64mf2 vl) {
    VLType<impl::vuint64mf2_t> vl_delegate(vl);
    return rv_add(x, y, vl_delegate);
}
inline impl::vuint64mf2_t rv_vle(uint64_t const* p, VLType_u64mf2 vl) {
    VLType<impl::vuint64mf2_t> vl_delegate(vl);
    return rv_vle(p, vl_delegate);
}
inline void rv_vse(uint64_t* p, impl::vuint64mf2_t v, VLType_u64mf2 vl) {
    VLType<impl::vuint64mf2_t> vl_delegate(vl);
    return rv_vse(p, v, vl_delegate);
}

};  // namespace rvv

// example usage:


void simplecopy(uint8_t* out, uint8_t const* in, std::size_t count) {
    constexpr rvv::LMUL vlmul = rvv::LMUL_m4;
    while (count > 0) {
        auto vl = rvv::rv_setvl<decltype(*out), vlmul>(count);
        rvv::rv_vse(out, rvv::rv_vle(in, vl), vl);
        in += vl;
        out += vl;
        count -= vl;
    }
}

void use_u64mf2(uint64_t* out, uint64_t const* in, std::size_t count) {
    while (count > 0) {
        auto vl = rvv::rv_setvl<uint64_t, rvv::LMUL_mf2>(count);
        auto outdata = rvv::rv_vle(out, vl);
        auto indata = rvv::rv_vle(in, vl);
        auto sum = rvv::rv_add(outdata, indata, vl);
        rvv::rv_vse(out, sum, vl);
        in += vl;
        out += vl;
        count -= vl;
    }
}


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
void bits_and_pieces(int32_t* out, int32_t const* in, std::size_t count) {
    constexpr rvv::LMUL vlmul = rvv::LMUL_mf2;
    while (count > 0) {
        auto vl = rvv::rv_setvl<decltype(*out), vlmul>(count);
        auto payload = rvv::rv_vle(in, vl);

        auto vlstuff = rvv::rv_setvl_u8m4(32);
        auto stuff = rvv::rv_dup(count, vlstuff);

        //rvv::VLType<uint32_t, rvv::LMUL_mf4> test = rvv::VLType_u64mf2(10).n();
        auto nxtest = rvv::VLType<uint32_t, rvv::LMUL_mf4>(10);
        auto wxtest = nxtest.w();
        rvv::VLType<uint64_t, rvv::LMUL_mf2> xtest = wxtest;
//        auto yytest = rvv::VLType<uint64_t, rvv::LMUL_mf4>(10).w();
        size_t foo = rvv::rv_meta<decltype(xtest)::lane_type, xtest.lmul>::min_lanes;

        stuff = rvv::rv_add(stuff, stuff, vlstuff);
//        stuff = rvv::rv_add(stuff, stuff, vl);             // Forbidden
        stuff = rvv::rv_add(stuff, stuff, vl * 4);
        stuff = rvv::rv_max(stuff, stuff, int(vl));
        stuff = rvv::rv_add(stuff, stuff, int(vl));
        auto wpay = rvv::rv_wadd(payload, payload, vl);
        payload = rvv::rv_nshr(wpay, 1, vl.w());
        stuff = rvv::rv_add(stuff, stuff, int(vl));
        stuff = rvv::rv_add(stuff, stuff, int(vl));
        stuff = rvv::rv_add(stuff, stuff, rvv::VLType_u8m4(vl));  // so formal!
        stuff = rvv::rv_add(stuff, stuff, 999);

        auto stuff_i32m4 = rvv::rv_reinterpret<int32_t>(stuff);
//        auto stuff_i32m4 = __riscv_reinterpret_i32m4(__riscv_vreinterpret_u32m4(stuff));
        auto stuff_i32m1 = __riscv_vget_i32m1(stuff_i32m4, 0);
        auto stuff_i32mf2 = __riscv_vlmul_trunc_i32mf2(stuff_i32m4);

        payload = rvv::rv_add(payload, stuff_i32mf2, vl);
//        payload = rvv::rv_add(payload, stuff_i32m1, vl);
//        payload = rvv::rv_add(payload, stuff_i32m4, vl);
        payload = rvv::rv_add(payload, payload, vl);
        rvv::rv_vse(out, payload, vl);
        out += vl;
        in += vl;
        count -= vl;
    }
}
#pragma GCC diagnostic pop
