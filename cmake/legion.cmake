option(ENABLE_LEGION "Enable Legion" OFF)

if(ENABLE_LEGION)
  find_package(Legion REQUIRED)
  list(APPEND TPL_LIBRARIES Legion::Legion)
endif()
