{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    ffmpeg
    (python3.withPackages (ps: with ps; [
      librosa
      numpy
      scipy
    ]))
  ];
}
