{
  description = "Voxel ray tracing native module for DimOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    dimos-rust = { url = "path:../../../../native/rust"; flake = false; };
  };

  outputs = { self, nixpkgs, flake-utils, dimos-rust }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        src = pkgs.runCommand "voxel-ray-tracing-src" {} ''
          mkdir -p $out/dimos/mapping/ray_tracing/rust
          cp -r ${./src} $out/dimos/mapping/ray_tracing/rust/src
          cp ${./Cargo.toml} $out/dimos/mapping/ray_tracing/rust/Cargo.toml
          cp ${./Cargo.lock} $out/dimos/mapping/ray_tracing/rust/Cargo.lock

          mkdir -p $out/native/rust
          cp -r ${dimos-rust}/dimos-module $out/native/rust/dimos-module
          cp -r ${dimos-rust}/dimos-module-macros $out/native/rust/dimos-module-macros
        '';
      in {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "voxel-ray-tracing";
          version = "0.1.0";

          inherit src;
          cargoRoot = "dimos/mapping/ray_tracing/rust";
          buildAndTestSubdir = "dimos/mapping/ray_tracing/rust";

          cargoHash = "sha256-0d0dlNDvDplA7oWTyUWOCOlS74Zie8uMQ+ps6lXntOI=";

          meta.mainProgram = "voxel_ray_tracing";
        };
      });
}
