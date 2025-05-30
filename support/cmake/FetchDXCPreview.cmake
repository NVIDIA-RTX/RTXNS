include(FetchContent)

if(!DXC_PREVIEW_VERSION)
	message(FATAL_ERROR "DXC_PREVIEW_VERSION not set")
endif()

FetchContent_Declare(
  dxc_preview
  URL https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DXC/${DXC_PREVIEW_VERSION}
  SOURCE_DIR "${DXC_PREVIEW_PATH}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  
FetchContent_MakeAvailable(dxc_preview)
 
set(DXC_PREVIEW_BIN_PATH "${dxc_preview_SOURCE_DIR}/build/native/bin/x64")
set(DXC_PREVIEW_PATH "${DXC_PREVIEW_BIN_PATH}/dxc.exe")
