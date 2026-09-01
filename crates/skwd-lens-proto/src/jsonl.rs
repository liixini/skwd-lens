use std::io::Write;

use serde::Serialize;

pub fn write_json_line(output: &mut impl Write, value: &impl Serialize) -> serde_json::Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n").map_err(serde_json::Error::io)
}

#[cfg(test)]
#[path = "jsonl_tests.rs"]
mod tests;
