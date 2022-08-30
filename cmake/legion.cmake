macro(flecsi_enable_legion)
  find_package(Legion REQUIRED)
  list(APPEND TPL_LIBRARIES Legion::Legion)
endmacro()
