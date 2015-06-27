#include "vw_exception.h"

namespace VW {

  vw_exception::vw_exception(const char* pfile, int plineNumber, std::string pmessage)
    : file(pfile), lineNumber(plineNumber), message(pmessage) {
  }

  vw_exception::vw_exception(const vw_exception& ex)
    : file(ex.file), lineNumber(ex.lineNumber), message(ex.message) {
  }

  vw_exception::~vw_exception() _NOEXCEPT {
  }

  const char* vw_exception::what() const _NOEXCEPT {
    return message.c_str();
  }

  const char* vw_exception::Filename() const {
    return file;
  }

  int vw_exception::LineNumber() const {
    return lineNumber;
  }
}
