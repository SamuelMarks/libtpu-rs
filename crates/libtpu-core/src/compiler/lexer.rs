//! `StableHLO` MLIR and XLA HLO lexer, recursive-descent parser, and Mosaic metadata extractor.

use super::CompiledProgram;
use super::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::HloModuleProto;
use crate::types::{ElementType, MemoryLayout, Shape};
use winnow::Parser;
use winnow::ascii::{digit1, multispace0};
use winnow::combinator::{alt, delimited, preceded, repeat, separated};
use winnow::error::ContextError;
use winnow::token::{any, take_till, take_while};

/// Lowered module components: `(name, root_opcode, parameter_shapes, output_shapes, instruction_bundles, output_slots)`.
pub(super) type LoweredModuleTuple = (
    String,
    TpuOpCode,
    Vec<Shape>,
    Vec<Shape>,
    Vec<TpuInstructionBundle>,
    Vec<usize>,
);

/// 1-based line, column, and byte offset span of a lexed [`HloToken`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenSpan {
    /// 1-based starting source line number.
    pub line: usize,
    /// 1-based starting source column number.
    pub column: usize,
    /// 0-based byte offset in the input source string.
    pub offset: usize,
    /// 1-based ending source line number.
    pub end_line: usize,
    /// 1-based ending source column number.
    pub end_column: usize,
    /// Byte length of the lexed token in the input source string.
    pub length: usize,
}

impl TokenSpan {
    /// Constructs a new [`TokenSpan`] from start coordinates, end coordinates, byte offset, and byte length.
    ///
    /// # Arguments
    ///
    /// * `line` - 1-based starting line number.
    /// * `column` - 1-based starting column number.
    /// * `offset` - 0-based byte offset in the source string.
    /// * `end_line` - 1-based ending line number.
    /// * `end_column` - 1-based ending column number.
    /// * `length` - Byte length of the token.
    #[must_use]
    pub const fn new(
        line: usize,
        column: usize,
        offset: usize,
        end_line: usize,
        end_column: usize,
        length: usize,
    ) -> Self {
        Self {
            line,
            column,
            offset,
            end_line,
            end_column,
            length,
        }
    }
}

/// A lexed [`HloToken`] paired with its source [`TokenSpan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpannedToken {
    /// Lexed token payload.
    pub token: HloToken,
    /// Source location span.
    pub span: TokenSpan,
}

/// Token produced by [`HloLexer`] when lexing `StableHLO` MLIR, `mhlo`, `chlo`, `sdy`, `tpu` Mosaic, or XLA HLO text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HloToken {
    /// Identifier, keyword, or dotted mnemonic (e.g., `module`, `func.func`, `stablehlo.add`, `true`, `false`, `nan`, `inf`).
    Ident(String),
    /// SSA value identifier starting with `%`, including optional multi-result `#idx` or `:count` suffix (e.g., `%arg0`, `%0#1`, `%res:2`).
    SsaValue(String),
    /// Module, function, or mesh symbol starting with `@` (e.g., `@main`, `@mesh`).
    Symbol(String),
    /// MLIR attribute alias or dialect attribute prefix starting with `#` (e.g., `#stablehlo`, `#sdy`, `#tpu`, `#loc0`).
    AttrAlias(String),
    /// MLIR type alias or dialect type prefix starting with `!` (e.g., `!stablehlo.token`, `!tpu.semaphore`, `!quant.uniform`).
    TypeAlias(String),
    /// MLIR basic block caret identifier starting with `^` (e.g., `^bb0`, `^entry`).
    CaretId(String),
    /// Unquoted numeric literal token, supporting integers, floats, scientific notation (`1.0e-5`), hex (`0xFF`, `0x1.8p+1`), and negative values.
    Number(String),
    /// Decoded string literal enclosed in double quotes (`"..."`), with escape sequences resolved.
    StringLiteral(String),
    /// Decoded binary byte payload from an MLIR hex-encoded string literal (`"0x0000803F..."`).
    HexBlob(Vec<u8>),
    /// Single-character delimiter or punctuation character (`=`, `:`, `,`, `(`, `)`, `{`, `}`, `[`, `]`, `<`, `>`, `?`, `*`, `+`, `-`, `/`).
    Punct(char),
    /// Arrow token (`->`).
    Arrow,
    /// Ellipsis token (`...`).
    Ellipsis,
    /// Nested symbol scope separator (`::`).
    DoubleColon,
    /// Less-than-or-equal comparison operator (`<=`).
    Le,
    /// Greater-than-or-equal comparison operator (`>=`).
    Ge,
    /// Equality comparison operator (`==`).
    EqEq,
    /// Inequality comparison operator (`!=`).
    NotEq,
}

/// Lexer for `StableHLO` MLIR (`stablehlo`, `mhlo`, `chlo`, `sdy`, `tpu` Mosaic) and XLA HLO textual modules powered by `winnow`.
#[derive(Debug, Clone, Default)]
pub struct HloLexer;

/// Skips leading whitespace, `//` single-line comments, and `/* ... */` block comments in `input`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference.
///
/// # Errors
///
/// Infallible (`Ok(())`), conforming to `winnow` parser signature.
#[allow(clippy::unnecessary_wraps)]
fn skip_whitespace_and_comments(input: &mut &str) -> Result<(), ContextError> {
    loop {
        *input = input.trim_start();
        if let Some(after_line_comment) = input.strip_prefix("//") {
            if let Some(nl_pos) = after_line_comment.find('\n') {
                *input = after_line_comment.get(nl_pos + 1..).unwrap_or("");
            } else {
                *input = "";
                return Ok(());
            }
        } else if let Some(after_block_start) = input.strip_prefix("/*") {
            if let Some(end_pos) = after_block_start.find("*/") {
                *input = after_block_start.get(end_pos + 2..).unwrap_or("");
            } else {
                *input = "";
                return Ok(());
            }
        } else {
            return Ok(());
        }
    }
}

/// Parses a signed, unsigned, scientific, or hexadecimal numeric literal (stopping before `->`) from `input`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` does not begin with a valid numeric literal.
fn winnow_lex_number(input: &mut &str) -> Result<HloToken, ContextError> {
    if let Some(rest) = input.strip_prefix("-inf")
        && !rest
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
    {
        *input = rest;
        return Ok(HloToken::Number(String::from("-inf")));
    }
    if let Some(rest) = input.strip_prefix("+inf")
        && !rest
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
    {
        *input = rest;
        return Ok(HloToken::Number(String::from("inf")));
    }
    let bytes = input.as_bytes();
    let mut idx = 0_usize;
    let mut has_leading_plus = false;
    let b0 = bytes.first().copied().unwrap_or(0);
    if b0 == b'-' || b0 == b'+' {
        if !bytes.get(1).is_some_and(u8::is_ascii_digit) {
            return Err(ContextError::new());
        }
        has_leading_plus = b0 == b'+';
        idx = 1;
    } else if !b0.is_ascii_digit() {
        return Err(ContextError::new());
    }

    let is_hex = bytes.get(idx).copied() == Some(b'0')
        && matches!(bytes.get(idx + 1).copied(), Some(b'x' | b'X'));
    if is_hex {
        idx += 2;
    }
    while idx < bytes.len() {
        let b = bytes.get(idx).copied().unwrap_or(0);
        let is_num_char = b.is_ascii_digit()
            || b == b'.'
            || (is_hex && (b.is_ascii_hexdigit() || matches!(b, b'p' | b'P')))
            || (!is_hex && matches!(b, b'e' | b'E'))
            || ((b == b'+' || b == b'-')
                && matches!(
                    bytes.get(idx.saturating_sub(1)).copied(),
                    Some(b'e' | b'E' | b'p' | b'P')
                )
                && bytes.get(idx + 1).copied() != Some(b'>'));
        if is_num_char {
            idx += 1;
        } else {
            break;
        }
    }
    let slice = input.get(..idx).unwrap_or("");
    *input = input.get(idx..).unwrap_or("");
    let cleaned = if has_leading_plus {
        slice.strip_prefix('+').unwrap_or(slice)
    } else {
        slice
    };
    Ok(HloToken::Number(String::from(cleaned)))
}

/// Parses an identifier or hyphenated HLO opcode (stopping before `->`) from `input`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` does not begin with an ASCII letter or `_`.
fn winnow_lex_ident(input: &mut &str) -> Result<HloToken, ContextError> {
    let bytes = input.as_bytes();
    let first = bytes.first().copied().unwrap_or(0);
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return Err(ContextError::new());
    }
    let mut len = 1_usize;
    while len < bytes.len() {
        let b = bytes.get(len).copied().unwrap_or(0);
        if b == b'-' && bytes.get(len + 1).copied() == Some(b'>') {
            break;
        }
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-') {
            len += 1;
        } else {
            break;
        }
    }
    let ident_str = input.get(..len).unwrap_or("");
    *input = input.get(len..).unwrap_or("");
    Ok(HloToken::Ident(String::from(ident_str)))
}

/// Decodes a hex digit character (`0..=9`, `a..=f`, `A..=F`) into its `0..=15` numeric value.
///
/// # Arguments
///
/// * `ch` - ASCII hex digit character.
#[must_use]
const fn hex_char_to_nibble(ch: char) -> Option<u8> {
    match ch {
        '0'..='9' => Some((ch as u8) - b'0'),
        'a'..='f' => Some((ch as u8) - b'a' + 10),
        'A'..='F' => Some((ch as u8) - b'A' + 10),
        _ => None,
    }
}

/// Parses a quoted string literal `"..."` (or hex blob `"0x..."`) from `input`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference starting with `"`.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` does not start with a valid double-quoted literal.
fn winnow_lex_string_or_hex(input: &mut &str) -> Result<HloToken, ContextError> {
    let Some(after_open) = input.strip_prefix('"') else {
        return Err(ContextError::new());
    };
    let mut chars = after_open.char_indices();
    let mut decoded = String::new();
    let mut end_byte_idx = None;
    let mut had_escape = false;
    while let Some((idx, ch)) = chars.next() {
        if ch == '"' {
            end_byte_idx = Some(idx);
            break;
        }
        if ch == '\\' {
            had_escape = true;
            if let Some((_, esc)) = chars.next() {
                match esc {
                    '"' => decoded.push('"'),
                    '\\' => decoded.push('\\'),
                    'n' => decoded.push('\n'),
                    't' => decoded.push('\t'),
                    'r' => decoded.push('\r'),
                    '0' => decoded.push('\0'),
                    'x' => {
                        let hi = chars.next().and_then(|(_, c)| hex_char_to_nibble(c));
                        let lo = chars.next().and_then(|(_, c)| hex_char_to_nibble(c));
                        if let (Some(h), Some(l)) = (hi, lo) {
                            decoded.push(char::from((h << 4) | l));
                        } else {
                            return Err(ContextError::new());
                        }
                    }
                    other => decoded.push(other),
                }
            } else {
                return Err(ContextError::new());
            }
        } else {
            decoded.push(ch);
        }
    }
    let Some(close_idx) = end_byte_idx else {
        return Err(ContextError::new());
    };
    *input = after_open.get(close_idx + 1..).unwrap_or("");
    if !had_escape
        && let Some(hex_body) = decoded
            .strip_prefix("0x")
            .or_else(|| decoded.strip_prefix("0X"))
        && !hex_body.is_empty()
        && hex_body.len() % 2 == 0
        && hex_body.chars().all(|c| c.is_ascii_hexdigit())
    {
        let mut bytes = Vec::with_capacity(hex_body.len() / 2);
        let hex_bytes = hex_body.as_bytes();
        let mut i = 0_usize;
        while i + 1 < hex_bytes.len() {
            let hi = hex_char_to_nibble(char::from(hex_bytes.get(i).copied().unwrap_or(b'0')))
                .unwrap_or(0);
            let lo = hex_char_to_nibble(char::from(hex_bytes.get(i + 1).copied().unwrap_or(b'0')))
                .unwrap_or(0);
            bytes.push((hi << 4) | lo);
            i += 2;
        }
        return Ok(HloToken::HexBlob(bytes));
    }
    Ok(HloToken::StringLiteral(decoded))
}

/// Parses a single [`HloToken`] from the front of `input` using `winnow` combinators.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference for `winnow`.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` is empty after skipping whitespace and comments.
#[allow(clippy::too_many_lines)]
fn winnow_lex_one_token(input: &mut &str) -> Result<HloToken, ContextError> {
    let _ = skip_whitespace_and_comments(input);
    if input.is_empty() {
        return Err(ContextError::new());
    }
    alt((
        alt((
            winnow_lex_string_or_hex,
            "->".value(HloToken::Arrow),
            "...".value(HloToken::Ellipsis),
            "::".value(HloToken::DoubleColon),
            "<=".value(HloToken::Le),
            ">=".value(HloToken::Ge),
            "==".value(HloToken::EqEq),
            "!=".value(HloToken::NotEq),
        )),
        winnow_lex_number,
        alt((
            preceded(
                '%',
                take_while(
                    0..,
                    |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.' | '#' | '-'),
                ),
            )
            .map(|name: &str| HloToken::SsaValue(format!("%{name}"))),
            preceded(
                '@',
                take_while(
                    0..,
                    |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.' | '-'),
                ),
            )
            .map(|name: &str| HloToken::Symbol(String::from(name))),
            preceded(
                '#',
                take_while(
                    1..,
                    |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.' | '-'),
                ),
            )
            .map(|name: &str| HloToken::AttrAlias(format!("#{name}"))),
            preceded(
                '!',
                take_while(
                    1..,
                    |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.' | '-'),
                ),
            )
            .map(|name: &str| HloToken::TypeAlias(format!("!{name}"))),
            preceded(
                '^',
                take_while(
                    1..,
                    |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.' | '-'),
                ),
            )
            .map(|name: &str| HloToken::CaretId(format!("^{name}"))),
            winnow_lex_ident,
            any.map(HloToken::Punct),
        )),
    ))
    .parse_next(input)
}

/// Extracts the inner body of a `tensor<...>` specification using `winnow`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference for `winnow`.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` does not begin with `tensor<...>`.
fn winnow_extract_tensor_body<'a>(input: &mut &'a str) -> Result<&'a str, ContextError> {
    *input = input.trim_start();
    let body = delimited("tensor<", take_till(1.., |c: char| c == '>'), '>').parse_next(input)?;
    *input = "";
    Ok(body.trim())
}

/// Parses an XLA HLO `<dtype>[<dims>]` shape specification using `winnow`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference for `winnow`.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` is not a valid `<dtype>[<dims>]` specification.
fn winnow_parse_hlo_shape_parts<'a>(
    input: &mut &'a str,
) -> Result<(&'a str, Vec<i64>), ContextError> {
    *input = input.trim_start();
    let dtype = take_till(1.., |c: char| c == '[').parse_next(input)?;
    let dims: Vec<i64> = delimited(
        '[',
        separated(
            0..,
            delimited(multispace0, digit1.try_map(str::parse::<i64>), multispace0),
            ',',
        ),
        ']',
    )
    .parse_next(input)?;
    *input = "";
    Ok((dtype.trim(), dims))
}

/// Computes 1-based `(line, column)` coordinates for a byte `offset` in `src`.
///
/// # Arguments
///
/// * `src` - Full source text.
/// * `offset` - 0-based byte offset within `src`.
#[must_use]
fn byte_offset_to_line_col(src: &str, offset: usize) -> (usize, usize) {
    let prefix = src.get(..offset).unwrap_or(src);
    let mut line = 1_usize;
    let mut col = 1_usize;
    for ch in prefix.chars() {
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += ch.len_utf8();
        }
    }
    (line, col)
}

impl HloLexer {
    /// Strictly tokenizes a `StableHLO` MLIR or XLA HLO source string into [`HloToken`]s using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `src` - Source text to tokenize.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HloLexerError`] if `src` is empty, contains an unclosed `/* ... */` block
    /// comment, contains an unclosed string literal, or has a malformed `\xHH` escape sequence.
    pub fn tokenize_checked(src: &str) -> LibTpuResult<Vec<HloToken>> {
        let trimmed = src.trim();
        if trimmed.is_empty() {
            return Err(LibTpuError::HloLexerError {
                offset: 0,
                snippet: String::new(),
                reason: String::from("Cannot lex empty HLO/StableHLO source string"),
            });
        }
        // Scan for unclosed comments, unclosed quotes, and invalid \xHH escapes while respecting // and /* */
        let bytes = src.as_bytes();
        let mut idx = 0_usize;
        while idx < bytes.len() {
            let b = bytes.get(idx).copied().unwrap_or(0);
            let next_b = bytes.get(idx + 1).copied().unwrap_or(0);
            if b == b'/' && next_b == b'/' {
                idx += 2;
                while idx < bytes.len() && bytes.get(idx).copied() != Some(b'\n') {
                    idx += 1;
                }
            } else if b == b'/' && next_b == b'*' {
                let comment_start = idx;
                idx += 2;
                let mut closed = false;
                while idx + 1 < bytes.len() {
                    if bytes.get(idx).copied() == Some(b'*')
                        && bytes.get(idx + 1).copied() == Some(b'/')
                    {
                        idx += 2;
                        closed = true;
                        break;
                    }
                    idx += 1;
                }
                if !closed {
                    return Err(LibTpuError::HloLexerError {
                        offset: comment_start,
                        snippet: String::from(src.get(comment_start..).unwrap_or(src)),
                        reason: String::from("Unclosed block comment '/*' in HLO/StableHLO source"),
                    });
                }
            } else if b == b'"' {
                let quote_start = idx;
                idx += 1;
                let mut closed = false;
                while idx < bytes.len() {
                    let ch = bytes.get(idx).copied().unwrap_or(0);
                    if ch == b'"' {
                        idx += 1;
                        closed = true;
                        break;
                    }
                    if ch == b'\\' {
                        idx += 1;
                        if idx >= bytes.len() {
                            break;
                        }
                        if bytes.get(idx).copied() == Some(b'x') {
                            let h1 = bytes
                                .get(idx + 1)
                                .copied()
                                .and_then(|c| hex_char_to_nibble(char::from(c)));
                            let h2 = bytes
                                .get(idx + 2)
                                .copied()
                                .and_then(|c| hex_char_to_nibble(char::from(c)));
                            if h1.is_none() || h2.is_none() {
                                return Err(LibTpuError::HloLexerError {
                                    offset: quote_start,
                                    snippet: String::from(src.get(quote_start..).unwrap_or(src)),
                                    reason: String::from(
                                        "Malformed '\\xHH' hex escape in string literal",
                                    ),
                                });
                            }
                            idx += 2;
                        }
                    }
                    idx += 1;
                }
                if !closed {
                    return Err(LibTpuError::HloLexerError {
                        offset: quote_start,
                        snippet: String::from(src.get(quote_start..).unwrap_or(src)),
                        reason: String::from("Unclosed string quote in HLO/StableHLO source"),
                    });
                }
            } else {
                idx += 1;
            }
        }
        Ok(Self::tokenize(src))
    }

    /// Tokenizes a `StableHLO` MLIR or XLA HLO source string into a sequence of [`HloToken`]s using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `src` - Source text to tokenize.
    #[must_use]
    pub fn tokenize(src: &str) -> Vec<HloToken> {
        let mut parser = terminated_token_stream();
        parser.parse(src).unwrap_or_default()
    }

    /// Tokenizes a `StableHLO` MLIR or XLA HLO source string into [`SpannedToken`]s with 1-based line,
    /// column, end-line, end-column, byte offset, and byte length spans using safe slice arithmetic.
    ///
    /// # Arguments
    ///
    /// * `src` - Source text to tokenize.
    #[must_use]
    pub fn tokenize_spanned(src: &str) -> Vec<SpannedToken> {
        let total_len = src.len();
        let mut rem = src;
        let mut out = Vec::new();
        loop {
            let _ = skip_whitespace_and_comments(&mut rem);
            let start_offset = total_len.saturating_sub(rem.len());
            let (line, column) = byte_offset_to_line_col(src, start_offset);
            if let Ok(token) = winnow_lex_one_token(&mut rem) {
                let end_offset = total_len.saturating_sub(rem.len());
                let length = end_offset.saturating_sub(start_offset);
                let (end_line, end_column) = byte_offset_to_line_col(src, end_offset);
                out.push(SpannedToken {
                    token,
                    span: TokenSpan::new(line, column, start_offset, end_line, end_column, length),
                });
            } else {
                break;
            }
        }
        out
    }
}

/// Builds a `winnow` parser that collects all [`HloToken`]s and consumes trailing whitespace and comments.
fn terminated_token_stream<'a>() -> impl Parser<&'a str, Vec<HloToken>, ContextError> {
    (
        repeat(0.., winnow_lex_one_token),
        skip_whitespace_and_comments,
    )
        .map(|(tokens, ())| tokens)
}

/// Recursive-descent parser for `StableHLO` MLIR and XLA HLO modules.
#[derive(Debug, Clone, Default)]
pub struct HloParser;

/// Structured representation of a parsed textual HLO or `StableHLO` MLIR module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedHloModule {
    /// Primary lowered TPU operation opcode.
    pub primary_op: TpuOpCode,
    /// Input parameter tensor shapes.
    pub parameter_shapes: Vec<Shape>,
    /// Output tensor shapes.
    pub output_shapes: Vec<Shape>,
    /// Lowered VLIW instruction bundle sequence.
    pub bundles: Vec<TpuInstructionBundle>,
    /// Register slot indices corresponding to each output tensor.
    pub output_slots: Vec<usize>,
}

/// Alias for [`ParsedHloModule`] representing the parsed HLO/StableHLO AST module.
pub type HloAstModule = ParsedHloModule;

impl HloParser {
    /// Strictly validates and parses a textual HLO or `StableHLO` MLIR module into a [`ParsedHloModule`].
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 program text to validate and parse.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HloLexerError`] or [`LibTpuError::HloParseError`] if `text` is empty,
    /// has unbalanced braces/parentheses/brackets, or contains an explicit `SYNTAX_ERROR` / `INVALID_HLO` token.
    pub fn parse_module(text: &str) -> LibTpuResult<ParsedHloModule> {
        let _checked = HloLexer::tokenize_checked(text)?;
        let spanned = HloLexer::tokenize_spanned(text);
        let mut depth = 0_i32;
        let mut last_line = 1_usize;
        let mut last_col = 1_usize;
        for st in &spanned {
            last_line = st.span.line;
            last_col = st.span.column;
            match &st.token {
                HloToken::Ident(id) if matches!(id.as_str(), "SYNTAX_ERROR" | "INVALID_HLO") => {
                    depth = -1;
                }
                HloToken::Punct('{' | '(' | '[') => depth += 1,
                HloToken::Punct('}' | ')' | ']') => depth -= 1,
                _ => {}
            }
            if depth < 0 {
                return Err(LibTpuError::HloParseError {
                    line: st.span.line,
                    column: st.span.column,
                    reason: String::from("Invalid or unbalanced HLO/StableHLO module syntax"),
                });
            }
        }
        if depth != 0 {
            return Err(LibTpuError::HloParseError {
                line: last_line,
                column: last_col,
                reason: String::from("Unclosed delimiter in HLO/StableHLO module"),
            });
        }
        Ok(CompiledProgram::parse_textual_program(text.trim()))
    }

    /// Parses a textual MLIR (`stablehlo`, `mhlo`, `chlo`, `sdy`, `tpu` Mosaic) or XLA HLO module
    /// into a strongly-typed hierarchical [`super::ast::AstModule`].
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 program text to parse.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if lexing or recursive-descent parsing fails.
    pub fn parse_ast_module(text: &str) -> LibTpuResult<super::ast::AstModule> {
        super::hlo_parser::parse_unified_ast_module(text)
    }

    /// Parses and semantically verifies a textual MLIR or XLA HLO module into an [`super::ast::AstModule`].
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 program text to parse and verify.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if lexing, recursive-descent parsing, SSA resolution, or shape/arity verification fails.
    pub fn parse_verified_ast_module(text: &str) -> LibTpuResult<super::ast::AstModule> {
        let ast = super::hlo_parser::parse_unified_ast_module(text)?;
        super::hlo_parser::verify_ast_module_semantics(&ast)?;
        Ok(ast)
    }

    /// Extracts the module name from a token stream (`@name` or `HloModule name`).
    ///
    /// # Arguments
    ///
    /// * `tokens` - Tokenized module slice.
    #[must_use]
    pub fn parse_module_name(tokens: &[HloToken]) -> String {
        for (idx, tok) in tokens.iter().enumerate() {
            if let HloToken::Symbol(sym) = tok {
                if !sym.is_empty() {
                    return sym.clone();
                }
            }
            if let HloToken::Ident(kw) = tok {
                if kw == "HloModule" {
                    if let Some(HloToken::Ident(name)) = tokens.get(idx + 1) {
                        return name.clone();
                    }
                }
            }
        }
        String::from("tpu_module")
    }
}

/// Extracted JAX Mosaic TPU and MHLO target attributes from a textual MLIR module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MosaicTargetMetadata {
    /// Target TPU generation string (`tpu.compilation_target`, e.g., `"tpu_v5p"`).
    pub compilation_target: String,
    /// Target TPU core type (`tpu.core_type`, e.g., `"tensor_core"` or `"sparse_core"`).
    pub core_type: String,
    /// Layout mode annotation (`mhlo.layout_mode`, e.g., `"default"` or `"{1,0}"`).
    pub layout_mode: String,
    /// Sharding annotation (`mhlo.sharding`, e.g., `"{replicated}"`).
    pub sharding: String,
    /// Grid dimension semantics (`dimension_semantics`, e.g., `["parallel", "arbitrary"]`).
    pub dimension_semantics: Vec<String>,
    /// Explicit VMEM scratchpad allocation size in bytes (`scratch_vmem_bytes`).
    pub scratch_vmem_bytes: usize,
    /// Whether any Mosaic `tpu.*` dialect operations were detected.
    pub has_mosaic_ops: bool,
}

impl CompiledProgram {
    /// Extracts JAX Mosaic TPU (`tpu.*`) and `mhlo.*` target attributes from an MLIR module string.
    ///
    /// # Arguments
    ///
    /// * `text` - Source MLIR module text.
    #[must_use]
    pub fn parse_mosaic_metadata(text: &str) -> MosaicTargetMetadata {
        let mut meta = MosaicTargetMetadata {
            has_mosaic_ops: text.contains("tpu."),
            scratch_vmem_bytes: Self::parse_single_usize_attr(text, "scratch_vmem_bytes")
                .unwrap_or(0),
            ..MosaicTargetMetadata::default()
        };
        for (key, dst) in [
            ("tpu.compilation_target", &mut meta.compilation_target),
            ("tpu.core_type", &mut meta.core_type),
            ("mhlo.layout_mode", &mut meta.layout_mode),
            ("mhlo.sharding", &mut meta.sharding),
        ] {
            if let Some(val) = text
                .split_once(key)
                .and_then(|(_, after)| after.split_once('"'))
                .and_then(|(_, quoted)| quoted.split_once('"'))
                .map(|(v, _)| v)
            {
                *dst = String::from(val);
            }
        }
        if let Some(list_str) = text
            .split_once("dimension_semantics")
            .and_then(|(_, after)| after.split_once('['))
            .and_then(|(_, inner)| inner.split_once(']'))
            .map(|(s, _)| s)
        {
            meta.dimension_semantics = list_str
                .split(',')
                .map(|item| String::from(item.trim().trim_matches('"').trim_start_matches("#tpu.")))
                .collect();
        }
        meta
    }

    /// Parses a `tensor<...>` or HLO shape token (e.g., `tensor<2x4xf32>`, `tensor<8xs32>`) into a [`Shape`] using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `token` - Tensor type specification substring.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidShape`] if `token` is not a valid `tensor<...>` specification.
    pub fn parse_tensor_type(token: &str) -> LibTpuResult<Shape> {
        let trimmed =
            winnow_extract_tensor_body
                .parse(token)
                .map_err(|_| LibTpuError::InvalidShape {
                    reason: format!("Invalid MLIR tensor type syntax: '{token}'"),
                })?;
        let (dims_part, dtype_str) = match trimmed.rsplit_once('x') {
            Some((d, dt)) => (Some(d), dt),
            None => (None, trimmed),
        };
        let elem_type = match dtype_str {
            "f32" => ElementType::F32,
            "f64" => ElementType::F64,
            "bf16" => ElementType::Bf16,
            "f16" => ElementType::F16,
            "s32" => ElementType::S32,
            "s64" => ElementType::S64,
            "s16" => ElementType::S16,
            "s8" => ElementType::S8,
            "u32" => ElementType::U32,
            "u64" => ElementType::U64,
            "u16" => ElementType::U16,
            "u8" => ElementType::U8,
            "i1" | "pred" => ElementType::Pred,
            "c64" | "complex<f32>" => ElementType::C64,
            "c128" | "complex<f64>" => ElementType::C128,
            "f8e5m2" => ElementType::F8E5M2,
            "f8e4m3fn" => ElementType::F8E4M3Fn,
            "f8e4m3b11fnuz" => ElementType::F8E4M3B11Fnuz,
            "s4" | "i4" => ElementType::S4,
            "u4" | "ui4" => ElementType::U4,
            "s2" | "i2" => ElementType::S2,
            "u2" | "ui2" => ElementType::U2,
            _ => {
                return Err(LibTpuError::InvalidShape {
                    reason: format!("Unknown MLIR element type '{dtype_str}' in '{token}'"),
                });
            }
        };
        let mut dims = Vec::new();
        if let Some(dp) = dims_part {
            for ds in dp.split('x') {
                let parsed = ds.parse::<i64>().map_err(|_| LibTpuError::InvalidShape {
                    reason: format!("Invalid dimension '{ds}' in '{token}'"),
                })?;
                dims.push(parsed);
            }
        }
        Shape::new(elem_type, dims, None, None)
    }

    /// Lowers a decoded binary [`HloModuleProto`] into module name, [`TpuOpCode`], parameter shapes,
    /// output shapes, VLIW instruction bundles, and output register slots.
    ///
    /// # Arguments
    ///
    /// * `proto` - Decoded binary `HloModuleProto`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError`] if an instruction shape specifies an unsupported [`ElementType`].
    #[allow(clippy::too_many_lines)]
    pub(super) fn lower_from_hlo_proto(proto: &HloModuleProto) -> LibTpuResult<LoweredModuleTuple> {
        let mut param_shapes = Vec::new();
        let mut slot_shapes = Vec::new();
        let mut root_op = TpuOpCode::Identity;
        let mut elem_type = ElementType::F32;
        let mut dims = vec![4_i64];
        let mut bundles = Vec::new();
        let mut id_to_slot = std::collections::BTreeMap::new();
        let mut next_slot: usize = 0;
        let mut pc: u32 = 0;

        for comp in &proto.computations {
            for inst in &comp.instructions {
                if let Some(shape) = &inst.shape {
                    if shape.element_type != 0 {
                        elem_type = ElementType::from_raw(shape.element_type)?;
                    }
                    if !shape.dimensions.is_empty() {
                        dims.clone_from(&shape.dimensions);
                    }
                }
                let inst_shape = Shape {
                    element_type: elem_type,
                    dynamic_dims: vec![false; dims.len()],
                    layout: MemoryLayout::default_tpu_tiled(dims.len()),
                    dims: dims.clone(),
                };
                if inst.opcode == "parameter" {
                    id_to_slot.insert(inst.id, next_slot);
                    next_slot += 1;
                    slot_shapes.push(inst_shape.clone());
                    param_shapes.push(inst_shape);
                } else {
                    let (op, et, _, _) = Self::parse_opcode_mnemonic(&inst.opcode);
                    root_op = op;
                    elem_type = et;
                    let op_slots: Vec<usize> = if inst.operand_ids.is_empty() {
                        (0..op.arity()).collect()
                    } else {
                        inst.operand_ids
                            .iter()
                            .enumerate()
                            .map(|(idx, id)| id_to_slot.get(id).copied().unwrap_or(idx))
                            .collect()
                    };
                    let in_shapes: Vec<Shape> = op_slots
                        .iter()
                        .filter_map(|&s| slot_shapes.get(s).cloned())
                        .collect();
                    let out_slot = next_slot;
                    id_to_slot.insert(inst.id, out_slot);
                    next_slot += 1;
                    slot_shapes.push(inst_shape.clone());
                    bundles.push(TpuInstructionBundle {
                        pc_offset: pc,
                        opcode: op,
                        operand_slots: op_slots,
                        output_slot: out_slot,
                        sync_flag_id: pc + 1,
                        attrs: InstructionAttrs {
                            input_shapes: in_shapes,
                            output_shape: Some(inst_shape),
                            ..InstructionAttrs::default()
                        },
                    });
                    pc += 1;
                }
            }
        }

        if param_shapes.is_empty() {
            let arity = root_op.arity();
            for _ in 0..arity {
                param_shapes.push(Shape {
                    element_type: elem_type,
                    dynamic_dims: vec![false; dims.len()],
                    layout: MemoryLayout::default_tpu_tiled(dims.len()),
                    dims: dims.clone(),
                });
            }
        }
        let out_shape = Shape {
            element_type: elem_type,
            dynamic_dims: vec![false; dims.len()],
            layout: MemoryLayout::default_tpu_tiled(dims.len()),
            dims,
        };
        if bundles.is_empty() {
            bundles.push(TpuInstructionBundle {
                pc_offset: 0,
                opcode: root_op,
                operand_slots: (0..param_shapes.len()).collect(),
                output_slot: param_shapes.len(),
                sync_flag_id: 1,
                attrs: InstructionAttrs {
                    input_shapes: param_shapes.clone(),
                    output_shape: Some(out_shape.clone()),
                    ..InstructionAttrs::default()
                },
            });
        }
        let last_slot = bundles.last().map_or(param_shapes.len(), |b| b.output_slot);
        Ok((
            proto.name.clone(),
            root_op,
            param_shapes,
            vec![out_shape],
            bundles,
            vec![last_slot],
        ))
    }

    /// Extracts a module name from textual MLIR/HLO (`module @foo` or `HloModule foo`) or returns `"tpu_module"`.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 lossy program text.
    pub(super) fn extract_module_name(text: &str) -> String {
        let tokens = HloLexer::tokenize(text);
        HloParser::parse_module_name(&tokens)
    }

    /// Strictly parses a comma-separated list of `usize` integers following `key` inside `line`.
    ///
    /// # Arguments
    ///
    /// * `line` - Instruction source line.
    /// * `key` - Attribute key identifier to search for.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::HloParseError`] if `key` is present with malformed numeric tokens.
    pub fn parse_usize_list_attr_checked(line: &str, key: &str) -> LibTpuResult<Vec<usize>> {
        let tokens = HloLexer::tokenize(line);
        let Some(key_pos) = tokens
            .iter()
            .position(|t| matches!(t, HloToken::Ident(id) if id.contains(key)))
        else {
            return Ok(Vec::new());
        };
        let tail = tokens.get(key_pos + 1..).unwrap_or(&[]);
        let op = tail
            .iter()
            .position(|t| matches!(t, HloToken::Punct('[' | '{' | '<')))
            .unwrap_or(0);
        let cl = tail
            .iter()
            .position(|t| matches!(t, HloToken::Punct(']' | '}' | '>')))
            .unwrap_or(0);
        let Some(inner) = tail.get(op + 1..cl) else {
            return Err(LibTpuError::HloParseError {
                line: 1,
                column: 1,
                reason: format!("Malformed delimiter list for attribute '{key}'"),
            });
        };
        let start = inner
            .iter()
            .rposition(|t| matches!(t, HloToken::Punct(':')))
            .map_or(0, |p| p + 1);
        let mut out = Vec::new();
        for tok in inner.get(start..).unwrap_or(inner) {
            if let HloToken::Number(n) = tok {
                let parsed = n.parse::<usize>().map_err(|_| LibTpuError::HloParseError {
                    line: 1,
                    column: 1,
                    reason: format!("Invalid usize literal '{n}' in attribute '{key}'"),
                })?;
                out.push(parsed);
            }
        }
        Ok(out)
    }

    /// Parses a comma-separated list of `usize` integers following `key` inside `line` (e.g., `permutation = [1, 0]`).
    ///
    /// # Arguments
    ///
    /// * `line` - Instruction source line.
    /// * `key` - Attribute key substring to search for.
    #[must_use]
    pub fn parse_usize_list_attr(line: &str, key: &str) -> Vec<usize> {
        Self::parse_usize_list_attr_checked(line, key).unwrap_or_default()
    }

    /// Parses a single `usize` integer attribute following `key` inside `line` (e.g., `dimension = 1`).
    ///
    /// # Arguments
    ///
    /// * `line` - Instruction source line.
    /// * `key` - Attribute key substring to search for.
    #[must_use]
    pub fn parse_single_usize_attr(line: &str, key: &str) -> Option<usize> {
        let tokens = HloLexer::tokenize(line);
        let key_pos = tokens
            .iter()
            .position(|t| matches!(t, HloToken::Ident(id) if id == key))?;
        if tokens.get(key_pos + 1) != Some(&HloToken::Punct('=')) {
            return None;
        }
        match tokens.get(key_pos + 2) {
            Some(HloToken::Number(n)) => n.parse::<usize>().ok(),
            _ => None,
        }
    }

    /// Parses a textual XLA HLO shape token (e.g., `f32[2,3]{1,0}` or `s32[4]`) into a [`Shape`] using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `token` - HLO shape token.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidShape`] if `token` is not a valid `<dtype>[<dims>]` specification.
    pub fn parse_hlo_shape_token(token: &str) -> LibTpuResult<Shape> {
        let (dtype_str, dims) =
            winnow_parse_hlo_shape_parts
                .parse(token)
                .map_err(|_| LibTpuError::InvalidShape {
                    reason: format!("Invalid HLO shape token syntax: '{token}'"),
                })?;
        let elem_type = match dtype_str {
            "f32" => ElementType::F32,
            "f64" => ElementType::F64,
            "bf16" => ElementType::Bf16,
            "f16" => ElementType::F16,
            "s32" => ElementType::S32,
            "s64" => ElementType::S64,
            "s16" => ElementType::S16,
            "s8" => ElementType::S8,
            "u32" => ElementType::U32,
            "u64" => ElementType::U64,
            "u16" => ElementType::U16,
            "u8" => ElementType::U8,
            "pred" | "i1" => ElementType::Pred,
            "c64" => ElementType::C64,
            "c128" => ElementType::C128,
            _ => {
                return Err(LibTpuError::InvalidShape {
                    reason: format!("Unknown HLO element type '{dtype_str}' in '{token}'"),
                });
            }
        };
        Shape::new(elem_type, dims, None, None)
    }

    /// Parses instruction attributes (`dot_general` dims, `permutation`, `broadcast_dimensions`,
    /// `slice` bounds, `reduce` dims, `dense<...>` constants, and result shape) from an instruction line.
    ///
    /// # Arguments
    ///
    /// * `line` - Instruction source line.
    /// * `input_shapes` - Shapes of the resolved input operands.
    /// * `default_elem_type` - Fallback element type for constant packing.
    #[must_use]
    pub fn parse_instruction_attrs(
        line: &str,
        input_shapes: &[Shape],
        default_elem_type: ElementType,
    ) -> InstructionAttrs {
        let mut attrs = InstructionAttrs {
            input_shapes: input_shapes.to_vec(),
            ..InstructionAttrs::default()
        };
        attrs.dot_lhs_contracting_dims = Self::parse_usize_list_attr(line, "lhs_contracting_dim");
        attrs.dot_rhs_contracting_dims = Self::parse_usize_list_attr(line, "rhs_contracting_dim");
        attrs.dot_lhs_batch_dims = Self::parse_usize_list_attr(line, "lhs_batching_dim");
        attrs.dot_rhs_batch_dims = Self::parse_usize_list_attr(line, "rhs_batching_dim");

        attrs.permutation = Self::parse_usize_list_attr(line, "permutation");
        if attrs.permutation.is_empty() {
            attrs.permutation = Self::parse_usize_list_attr(line, "dims");
        }
        attrs.broadcast_dims = Self::parse_usize_list_attr(line, "broadcast_dimensions");
        attrs.slice_starts = Self::parse_usize_list_attr(line, "start_indices");
        attrs.slice_limits = Self::parse_usize_list_attr(line, "limit_indices");
        attrs.slice_strides = Self::parse_usize_list_attr(line, "strides");
        attrs.pad_edge_low = Self::parse_usize_list_attr(line, "edge_padding_low");
        attrs.pad_edge_high = Self::parse_usize_list_attr(line, "edge_padding_high");
        attrs.pad_interior = Self::parse_usize_list_attr(line, "interior_padding");
        attrs.reduce_dims = Self::parse_usize_list_attr(line, "dimensions");
        attrs.window_dims = Self::parse_usize_list_attr(line, "window_dimensions");
        attrs.window_strides = Self::parse_usize_list_attr(line, "window_strides");
        if let Some(cd) = Self::parse_single_usize_attr(line, "dimension") {
            attrs.concat_dim = cd;
        }
        if let Some(id) = Self::parse_single_usize_attr(line, "iota_dimension") {
            attrs.iota_dim = id;
        }
        for dir_pat in [
            "comparison_direction = #stablehlo<comparison_direction ",
            "direction=",
        ] {
            if let Some((_, after_dir)) = line.split_once(dir_pat) {
                let token_dir = after_dir
                    .split(|c: char| !c.is_ascii_alphabetic())
                    .find(|s| !s.is_empty())
                    .unwrap_or("");
                attrs.comparison_direction = String::from(token_dir);
            }
        }
        if let Some(target) = line
            .split_once("call_target_name")
            .and_then(|(_, after_cc)| after_cc.split_once('"'))
            .and_then(|(_, quoted)| quoted.split_once('"'))
            .map(|(t, _)| t)
        {
            attrs.custom_call_target = String::from(target);
        }

        // Extract output shape from `-> tensor<...>` or `: tensor<...>` or HLO `<type>[<dims>]`
        if let Some((_, seg)) = line.rsplit_once("tensor<") {
            attrs.output_shape = Self::parse_tensor_type(&format!("tensor<{seg}")).ok();
        }
        if attrs.output_shape.is_none() {
            for tok in line.split_whitespace() {
                if let Ok(sh) = Self::parse_hlo_shape_token(tok) {
                    attrs.output_shape = Some(sh);
                    break;
                }
            }
        }

        let elem_type = attrs
            .output_shape
            .as_ref()
            .map_or(default_elem_type, |s| s.element_type);
        if let Some((_, after_dense)) = line.split_once("dense<") {
            let inner = after_dense.split('>').next().unwrap_or(after_dense);
            let cleaned: String = inner
                .chars()
                .map(|c| if c == '[' || c == ']' { ' ' } else { c })
                .collect();
            let vals: Vec<f64> = cleaned
                .split(',')
                .filter_map(|t| t.trim().parse::<f64>().ok())
                .collect();
            let count = attrs
                .output_shape
                .as_ref()
                .map_or(vals.len(), Shape::element_count)
                .max(vals.len());
            let expanded = if vals.len() == 1 && count > 1 {
                vec![vals.first().copied().unwrap_or(0.0); count]
            } else {
                vals
            };
            attrs.constant_bytes = elem_type.pack_from_f64_slice(&expanded);
        }
        attrs
    }

    /// Parses an instruction line or mnemonic token into `(TpuOpCode, ElementType, arity, default_dims)`.
    ///
    /// # Arguments
    ///
    /// * `token` - Instruction line or operation mnemonic.
    #[allow(clippy::too_many_lines)]
    pub(super) fn parse_opcode_mnemonic(token: &str) -> (TpuOpCode, ElementType, usize, Vec<i64>) {
        let lower = token.to_ascii_lowercase();
        let table: &[(&str, TpuOpCode, ElementType, usize, &[i64])] = &[
            ("s32_add", TpuOpCode::AddS32, ElementType::S32, 2, &[4]),
            ("mul_f32", TpuOpCode::MulF32, ElementType::F32, 2, &[4]),
            ("multiply", TpuOpCode::MulF32, ElementType::F32, 2, &[4]),
            ("sub_f32", TpuOpCode::SubF32, ElementType::F32, 2, &[4]),
            ("subtract", TpuOpCode::SubF32, ElementType::F32, 2, &[4]),
            ("div_f32", TpuOpCode::DivF32, ElementType::F32, 2, &[4]),
            ("divide", TpuOpCode::DivF32, ElementType::F32, 2, &[4]),
            ("max_f32", TpuOpCode::MaxF32, ElementType::F32, 2, &[4]),
            ("maximum", TpuOpCode::MaxF32, ElementType::F32, 2, &[4]),
            ("min_f32", TpuOpCode::MinF32, ElementType::F32, 2, &[4]),
            ("minimum", TpuOpCode::MinF32, ElementType::F32, 2, &[4]),
            (
                "dot_general",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul_push_rhs",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul_acc",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul_pop",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.enqueue_dma",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.wait_dma",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.dma_copy",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.dma_wait",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.semaphore_signal",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.semaphore_wait",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.semaphore_read",
                TpuOpCode::CustomCall,
                ElementType::S32,
                1,
                &[4],
            ),
            (
                "tpu.barrier",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.vector_load",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.vector_store",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tpu.load", TpuOpCode::Identity, ElementType::F32, 1, &[4]),
            ("tpu.store", TpuOpCode::Identity, ElementType::F32, 1, &[4]),
            ("tpu.iota", TpuOpCode::Iota, ElementType::S32, 0, &[4]),
            (
                "tpu.repeat",
                TpuOpCode::BroadcastInDim,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tpu.rotate", TpuOpCode::Reverse, ElementType::F32, 1, &[4]),
            (
                "tpu.pack_subelements",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.unpack_subelements",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.memref_squeeze",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.memref_bitcast",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.relabel",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tpu.bitcast", TpuOpCode::Reshape, ElementType::F32, 1, &[4]),
            ("tpu.roll", TpuOpCode::Reverse, ElementType::F32, 1, &[4]),
            (
                "tpu.memref_slice",
                TpuOpCode::Slice,
                ElementType::F32,
                1,
                &[2],
            ),
            (
                "tpu.erase_memref_layout",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.region",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "collective_permute",
                TpuOpCode::CollectivePermute,
                ElementType::F32,
                1,
                &[4],
            ),
            ("all_to_all", TpuOpCode::AllToAll, ElementType::F32, 1, &[4]),
            (
                "exponential_minus_one",
                TpuOpCode::Expm1,
                ElementType::F32,
                1,
                &[4],
            ),
            ("expm1", TpuOpCode::Expm1, ElementType::F32, 1, &[4]),
            ("log_plus_one", TpuOpCode::Log1p, ElementType::F32, 1, &[4]),
            ("log1p", TpuOpCode::Log1p, ElementType::F32, 1, &[4]),
            ("logistic", TpuOpCode::Logistic, ElementType::F32, 1, &[4]),
            ("erf", TpuOpCode::Erf, ElementType::F32, 1, &[4]),
            (
                "round_nearest_even",
                TpuOpCode::RoundNearestEven,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "round_nearest_afz",
                TpuOpCode::RoundNearestAfz,
                ElementType::F32,
                1,
                &[4],
            ),
            ("sign", TpuOpCode::Sign, ElementType::F32, 1, &[4]),
            ("is_finite", TpuOpCode::IsFinite, ElementType::Pred, 1, &[4]),
            ("cbrt", TpuOpCode::Cbrt, ElementType::F32, 1, &[4]),
            ("tan(", TpuOpCode::Tan, ElementType::F32, 1, &[4]),
            ("stablehlo.tan", TpuOpCode::Tan, ElementType::F32, 1, &[4]),
            ("power", TpuOpCode::Pow, ElementType::F32, 2, &[4]),
            ("atan2", TpuOpCode::Atan2, ElementType::F32, 2, &[4]),
            ("clamp", TpuOpCode::Clamp, ElementType::F32, 3, &[4]),
            (
                "popcnt",
                TpuOpCode::PopulationCount,
                ElementType::S32,
                1,
                &[4],
            ),
            ("pad", TpuOpCode::Pad, ElementType::F32, 2, &[4]),
            (
                "select_and_scatter",
                TpuOpCode::Scatter,
                ElementType::F32,
                3,
                &[4],
            ),
            (
                "batch_norm_inference",
                TpuOpCode::MulF32,
                ElementType::F32,
                2,
                &[4],
            ),
            ("fft", TpuOpCode::Fft, ElementType::F32, 1, &[4]),
            (
                "cholesky",
                TpuOpCode::Cholesky,
                ElementType::F32,
                1,
                &[2, 2],
            ),
            (
                "triangular_solve",
                TpuOpCode::TriangularSolve,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            ("top_k", TpuOpCode::TopK, ElementType::F32, 1, &[4]),
            (
                "uniform_quantize",
                TpuOpCode::UniformQuantize,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "uniform_dequantize",
                TpuOpCode::UniformDequantize,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "partition_id",
                TpuOpCode::Constant,
                ElementType::U32,
                0,
                &[1],
            ),
            ("replica_id", TpuOpCode::Constant, ElementType::U32, 0, &[1]),
            (
                "after_all",
                TpuOpCode::Identity,
                ElementType::Token,
                1,
                &[1],
            ),
            ("infeed", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            ("outfeed", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            ("send", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            ("recv", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            (
                "get_tuple_element",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tuple", TpuOpCode::Identity, ElementType::Tuple, 1, &[4]),
            (
                " dot(",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            ("neg_f32", TpuOpCode::NegF32, ElementType::F32, 1, &[4]),
            ("negate", TpuOpCode::NegF32, ElementType::F32, 1, &[4]),
            ("exp_f32", TpuOpCode::ExpF32, ElementType::F32, 1, &[4]),
            ("exponential", TpuOpCode::ExpF32, ElementType::F32, 1, &[4]),
            ("tanh_f32", TpuOpCode::TanhF32, ElementType::F32, 1, &[4]),
            ("tanh", TpuOpCode::TanhF32, ElementType::F32, 1, &[4]),
            ("abs", TpuOpCode::Abs, ElementType::F32, 1, &[4]),
            ("log", TpuOpCode::Log, ElementType::F32, 1, &[4]),
            ("rsqrt", TpuOpCode::Rsqrt, ElementType::F32, 1, &[4]),
            ("sqrt", TpuOpCode::Sqrt, ElementType::F32, 1, &[4]),
            ("cosine", TpuOpCode::Cosine, ElementType::F32, 1, &[4]),
            ("sine", TpuOpCode::Sine, ElementType::F32, 1, &[4]),
            ("floor", TpuOpCode::Floor, ElementType::F32, 1, &[4]),
            ("ceil", TpuOpCode::Ceil, ElementType::F32, 1, &[4]),
            ("not", TpuOpCode::Not, ElementType::Pred, 1, &[4]),
            ("remainder", TpuOpCode::Rem, ElementType::F32, 2, &[4]),
            (
                "compare_lt",
                TpuOpCode::CompareLt,
                ElementType::F32,
                2,
                &[4],
            ),
            (
                "compare_gt",
                TpuOpCode::CompareGt,
                ElementType::F32,
                2,
                &[4],
            ),
            (
                "compare_eq",
                TpuOpCode::CompareEq,
                ElementType::F32,
                2,
                &[4],
            ),
            (
                "shift_left",
                TpuOpCode::ShiftLeft,
                ElementType::S32,
                2,
                &[4],
            ),
            (
                "shift_right_arithmetic",
                TpuOpCode::ShiftRightArithmetic,
                ElementType::S32,
                2,
                &[4],
            ),
            (
                "shift_right_logical",
                TpuOpCode::ShiftRightLogical,
                ElementType::U32,
                2,
                &[4],
            ),
            (
                "population_count",
                TpuOpCode::PopulationCount,
                ElementType::S32,
                1,
                &[4],
            ),
            (
                "count_leading_zeros",
                TpuOpCode::CountLeadingZeros,
                ElementType::S32,
                1,
                &[4],
            ),
            (
                "convolution",
                TpuOpCode::Convolution,
                ElementType::F32,
                2,
                &[4],
            ),
            ("gather", TpuOpCode::Gather, ElementType::F32, 2, &[4]),
            ("scatter", TpuOpCode::Scatter, ElementType::F32, 3, &[4]),
            (
                "dynamic_update_slice",
                TpuOpCode::DynamicUpdateSlice,
                ElementType::F32,
                3,
                &[4],
            ),
            (
                "dynamic_slice",
                TpuOpCode::DynamicSlice,
                ElementType::F32,
                2,
                &[2],
            ),
            (
                "reduce_window",
                TpuOpCode::ReduceWindow,
                ElementType::F32,
                1,
                &[4],
            ),
            ("sort", TpuOpCode::Sort, ElementType::F32, 1, &[4]),
            (
                "rng_bit_generator",
                TpuOpCode::RngBitGenerator,
                ElementType::U32,
                1,
                &[4],
            ),
            ("select", TpuOpCode::Select, ElementType::F32, 3, &[4]),
            ("iota", TpuOpCode::Iota, ElementType::F32, 1, &[4]),
            ("reverse", TpuOpCode::Reverse, ElementType::F32, 1, &[4]),
            (
                "reduce_max",
                TpuOpCode::ReduceMax,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "reduce_min",
                TpuOpCode::ReduceMin,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "reduce_sum",
                TpuOpCode::ReduceSumF32,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "reduce_scatter",
                TpuOpCode::ReduceScatter,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "all_gather",
                TpuOpCode::AllGather,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "all_reduce",
                TpuOpCode::AllReduceSumF32,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "custom_call",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            ("xor_u8", TpuOpCode::XorBytes, ElementType::U8, 2, &[16]),
            (
                "bitcast_convert",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            ("convert", TpuOpCode::Convert, ElementType::F32, 1, &[4]),
            (
                "transpose",
                TpuOpCode::Transpose,
                ElementType::F32,
                1,
                &[2, 2],
            ),
            (
                "broadcast_in_dim",
                TpuOpCode::BroadcastInDim,
                ElementType::F32,
                1,
                &[4],
            ),
            ("slice", TpuOpCode::Slice, ElementType::F32, 1, &[2]),
            (
                "concatenate",
                TpuOpCode::Concatenate,
                ElementType::F32,
                2,
                &[4],
            ),
            ("constant", TpuOpCode::Constant, ElementType::F32, 0, &[4]),
            ("reshape", TpuOpCode::Reshape, ElementType::F32, 1, &[4]),
            ("and", TpuOpCode::And, ElementType::S32, 2, &[4]),
            ("or", TpuOpCode::Or, ElementType::S32, 2, &[4]),
            ("add", TpuOpCode::AddF32, ElementType::F32, 2, &[4]),
        ];
        let tokens = HloLexer::tokenize(&lower);
        let ident_tokens: Vec<&str> = tokens
            .iter()
            .filter_map(|t| match t {
                HloToken::Ident(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        for &(pat, op, et, arity, dims) in table {
            let clean_pat = pat.trim().trim_end_matches('(');
            let matched = ident_tokens.iter().any(|&id| {
                (id == clean_pat)
                    | (id.strip_prefix("stablehlo.") == Some(clean_pat))
                    | ((clean_pat.len() >= 4) & id.contains(clean_pat))
            });
            if matched {
                return (op, et, arity, dims.to_vec());
            }
        }
        (TpuOpCode::Identity, ElementType::F32, 1, vec![4])
    }

    /// Parses textual HLO or `StableHLO` MLIR into primary [`TpuOpCode`], parameter [`Shape`]s,
    /// output [`Shape`]s, topological [`TpuInstructionBundle`] SSA DAG, and output slot indices.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 lossy program text or HLO instruction opcode.
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    pub(super) fn parse_textual_program(text: &str) -> ParsedHloModule {
        if ["sdy.", "chlo.", "scf.for", "to_apply="]
            .iter()
            .any(|kw| text.contains(kw))
            && let Ok(lowered) =
                super::hlo_parser::parse_unified_ast_module(text).and_then(|ast_mod| {
                    if ast_mod.functions.is_empty() && ast_mod.hlo_computations.is_empty() {
                        Err(LibTpuError::HloParseError {
                            line: 1,
                            column: 1,
                            reason: String::from("Empty AST module"),
                        })
                    } else {
                        super::hlo_parser::lower_ast_module_to_parsed_hlo(&ast_mod)
                    }
                })
        {
            return lowered;
        }
        let (mut primary_op, mut elem_type, default_inputs, mut dims) =
            Self::parse_opcode_mnemonic(text);

        let mut explicit_param_shapes = Vec::new();
        let mut explicit_output_shapes = Vec::new();
        let mut ssa_slots = std::collections::BTreeMap::new();
        let mut slot_shapes: Vec<Shape> = Vec::new();
        let mut next_slot = 0_usize;

        // Extract `func.func @main(...) -> ...` signature shapes and SSA parameter names when present
        let _ = text
            .split_once("func.func")
            .and_then(|(_, after_func)| after_func.split_once('('))
            .and_then(|(_, after_open)| after_open.split_once(')'))
            .map(|(params_str, after_params)| {
                for param_decl in params_str.split(',') {
                    let _ = param_decl.split_once(':').map(|(var_part, type_part)| {
                        let _ = Self::parse_tensor_type(type_part.trim()).map(|sh| {
                            elem_type = sh.element_type;
                            dims.clone_from(&sh.dims);
                            ssa_slots.insert(String::from(var_part.trim()), next_slot);
                            next_slot += 1;
                            slot_shapes.push(sh.clone());
                            explicit_param_shapes.push(sh);
                        });
                    });
                }
                let _ = after_params.split_once("->").map(|(_, after_arrow)| {
                    let ret_header = after_arrow.split('{').next().unwrap_or(after_arrow);
                    for segment in ret_header.split("tensor<").skip(1) {
                        let candidate = format!("tensor<{segment}");
                        let _ = Self::parse_tensor_type(&candidate).map(|sh| {
                            explicit_output_shapes.push(sh);
                        });
                    }
                });
            });

        if explicit_param_shapes.is_empty() {
            for segment in text.split("tensor<").skip(1) {
                let candidate = format!("tensor<{segment}");
                if let Ok(sh) = Self::parse_tensor_type(&candidate) {
                    elem_type = sh.element_type;
                    dims = sh.dims;
                    break;
                }
            }
        }

        let mut bundles = Vec::new();
        let mut output_slots = Vec::new();
        for line in text.lines() {
            let trimmed_line = line.trim();
            if trimmed_line.starts_with("return ") || trimmed_line.starts_with("func.return ") {
                let before_colon = trimmed_line.split(':').next().unwrap_or(trimmed_line);
                for tok in before_colon.split(|c: char| c.is_whitespace() || c == ',') {
                    if let Some(&slot) = ssa_slots.get(tok.trim()) {
                        output_slots.push(slot);
                    }
                }
                continue;
            }
            if !trimmed_line.contains('=') || trimmed_line.starts_with("HloModule") {
                continue;
            }
            let (lhs, rhs) = trimmed_line.split_once('=').unwrap_or(("", trimmed_line));
            let lhs_var = String::from(lhs.split_whitespace().last().unwrap_or("").trim());
            if rhs.contains("parameter(") {
                for tok in rhs.split_whitespace() {
                    if let Ok(sh) = Self::parse_hlo_shape_token(tok) {
                        elem_type = sh.element_type;
                        dims.clone_from(&sh.dims);
                        ssa_slots.insert(lhs_var.clone(), next_slot);
                        next_slot += 1;
                        slot_shapes.push(sh.clone());
                        explicit_param_shapes.push(sh);
                    }
                }
                continue;
            }
            if let Some((_, after_tuple)) = rhs.split_once("tuple(") {
                let tuple_args = after_tuple.split(')').next().unwrap_or(after_tuple);
                output_slots.extend(
                    tuple_args
                        .split(|c: char| c.is_whitespace() || c == ',')
                        .filter_map(|tok| ssa_slots.get(tok.trim()).copied()),
                );
                explicit_output_shapes.extend(
                    output_slots
                        .iter()
                        .filter_map(|&slot| slot_shapes.get(slot).cloned()),
                );
                continue;
            }
            let (mut line_op, _, line_arity, _) = Self::parse_opcode_mnemonic(rhs);
            let mut op_slots = Vec::new();
            let rhs_before_colon = rhs.split(':').next().unwrap_or(rhs);
            for tok in rhs_before_colon
                .split(|c: char| c.is_whitespace() || c == ',' || c == '(' || c == ')')
            {
                let t = tok.trim();
                if !t.is_empty() {
                    if let Some(&slot) = ssa_slots.get(t) {
                        op_slots.push(slot);
                    }
                }
            }
            if op_slots.is_empty() {
                op_slots = (0..line_arity).collect();
            }
            let in_shapes: Vec<Shape> = op_slots
                .iter()
                .filter_map(|&s| slot_shapes.get(s).cloned())
                .collect();
            let attrs = Self::parse_instruction_attrs(rhs, &in_shapes, elem_type);
            let default_bundle_shape = Shape {
                element_type: elem_type,
                dynamic_dims: vec![false; dims.len()],
                layout: MemoryLayout::default_tpu_tiled(dims.len()),
                dims: dims.clone(),
            };
            let bundle_out_shape = attrs.output_shape.clone().unwrap_or(default_bundle_shape);
            if line_op == TpuOpCode::AddF32
                && matches!(
                    bundle_out_shape.element_type,
                    ElementType::S32
                        | ElementType::S64
                        | ElementType::S16
                        | ElementType::S8
                        | ElementType::U32
                        | ElementType::U64
                )
            {
                line_op = TpuOpCode::AddS32;
            }
            primary_op = line_op;
            let out_slot = next_slot;
            ssa_slots.insert(lhs_var, out_slot);
            next_slot += 1;
            slot_shapes.push(bundle_out_shape);
            let pc = u32::try_from(bundles.len()).unwrap_or(0);
            bundles.push(TpuInstructionBundle {
                pc_offset: pc,
                opcode: line_op,
                operand_slots: op_slots,
                output_slot: out_slot,
                sync_flag_id: pc + 1,
                attrs,
            });
        }

        if explicit_param_shapes.is_empty() {
            for _ in 0..default_inputs {
                explicit_param_shapes.push(Shape {
                    element_type: elem_type,
                    dynamic_dims: vec![false; dims.len()],
                    layout: MemoryLayout::default_tpu_tiled(dims.len()),
                    dims: dims.clone(),
                });
            }
        }
        if explicit_output_shapes.is_empty() {
            let fallback_out = Shape {
                element_type: elem_type,
                dynamic_dims: vec![false; dims.len()],
                layout: MemoryLayout::default_tpu_tiled(dims.len()),
                dims,
            };
            let inferred_out = bundles
                .last()
                .and_then(|b| b.attrs.output_shape.clone())
                .unwrap_or(fallback_out);
            explicit_output_shapes.push(inferred_out);
        }

        if bundles.is_empty() {
            let operand_slots: Vec<usize> = (0..explicit_param_shapes.len()).collect();
            let out_slot = explicit_param_shapes.len();
            bundles.push(TpuInstructionBundle {
                pc_offset: 0,
                opcode: primary_op,
                operand_slots,
                output_slot: out_slot,
                sync_flag_id: 1,
                attrs: InstructionAttrs {
                    input_shapes: explicit_param_shapes.clone(),
                    output_shape: explicit_output_shapes.first().cloned(),
                    ..InstructionAttrs::default()
                },
            });
        }
        if output_slots.is_empty() {
            let last_slot = bundles
                .last()
                .map_or(explicit_param_shapes.len(), |b| b.output_slot);
            output_slots.push(last_slot);
        }
        ParsedHloModule {
            primary_op,
            parameter_shapes: explicit_param_shapes,
            output_shapes: explicit_output_shapes,
            bundles,
            output_slots,
        }
    }
}
